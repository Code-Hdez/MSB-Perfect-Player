"""
Dolphin emulator input via virtual Xbox 360 controller (ViGEm).

Uses a **background pump thread** to send HID reports at ~120 Hz,
guaranteeing the virtual controller never appears idle to Dolphin /
XInput — even when the computer-vision main loop runs at only 10–25 FPS.

Reliability features
--------------------
  * Windows ``timeBeginPeriod(1)`` for ≤ 2 ms sleep resolution.
  * ``perf_counter``-based hybrid sleep + spin-wait for precise ticks.
  * Monotonic heartbeat on right trigger (4 cycling byte values 1→4) —
    no two consecutive HID reports are ever byte-identical.
  * Periodic **XInput device verification** (every 5 s) via
    ``XInputGetState`` — detects device-level disconnection even when
    ``vgamepad.update()`` does not raise.
  * Auto-recovery: recreates the virtual device on detected disconnect.
  * Extra strong-reference to the ViGEm bus singleton prevents GC.
  * Thread-safe state passing via GIL-protected simple assignments.

Architecture
------------
  Main thread (CV loop)                 Pump thread (daemon, ~120 Hz)
  ─────────────────────                 ─────────────────────────────
  apply(sx, sy, swing)  ──writes──►  _sx, _sy, _want_a  ──reads──►
                                      left_joystick_float(sx, sy)
                                      press/release A / B
                                      right_trigger heartbeat
                                      update()   ← guaranteed ~120×/sec

Setup (one-time)
----------------
1. ``pip install vgamepad``
   (first run auto-installs ViGEmBus driver — may need UAC).

2. Run ``bat_live.py`` first (so the virtual gamepad exists).

3. **Dolphin → Controllers → Port 1 → Standard Controller → Configure**:
   a. Device = **XInput/0/Gamepad**
   b. Click **Default** (auto-maps Left Stick → Main Stick, A → A, etc.)
   c. **Enable** *Background Input* (if present).
   d. Save the profile.

4. Start the game, then run ``bat_live.py``.

Usage::

    from msb.dolphin_input import DolphinGamepadInput

    inp = DolphinGamepadInput()
    inp.connect()
    inp.apply(0.7, 0.5, False)                # stick right, no swing
    inp.apply(0.5, 0.5, True)                 # centre, swing!
    inp.apply(0.5, 0.5, False, press_b=True)  # B held
    inp.safe_neutral()
    inp.close()
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as _wt
import sys
import threading
import time
from typing import Optional


# WINDOWS HIGH-RESOLUTION TIMER

_winmm = None
try:
    _winmm = ctypes.windll.winmm
except (AttributeError, OSError):
    pass


def _begin_hires_timer() -> None:
    """Request 1 ms timer resolution from Windows."""
    if _winmm is not None:
        _winmm.timeBeginPeriod(1)


def _end_hires_timer() -> None:
    """Restore default timer resolution."""
    if _winmm is not None:
        _winmm.timeEndPeriod(1)


# XINPUT DEVICE VERIFICATION

_xinput_dll = None
for _dll_name in ("xinput1_4", "xinput1_3", "xinput9_1_0"):
    try:
        _xinput_dll = ctypes.WinDLL(_dll_name + ".dll")
        break
    except OSError:
        continue


class _XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [
        ("wButtons",      ctypes.c_ushort),
        ("bLeftTrigger",  ctypes.c_ubyte),
        ("bRightTrigger", ctypes.c_ubyte),
        ("sThumbLX",      ctypes.c_short),
        ("sThumbLY",      ctypes.c_short),
        ("sThumbRX",      ctypes.c_short),
        ("sThumbRY",      ctypes.c_short),
    ]


class _XINPUT_STATE(ctypes.Structure):
    _fields_ = [
        ("dwPacketNumber", _wt.DWORD),
        ("Gamepad",        _XINPUT_GAMEPAD),
    ]


def _xinput_connected(slot: int) -> bool:
    """Return True if XInput slot (0–3) has a connected device."""
    if _xinput_dll is None:
        return True  # can't verify — optimistic fallback
    state = _XINPUT_STATE()
    rc = _xinput_dll.XInputGetState(slot, ctypes.byref(state))
    return rc == 0  # 0 = SUCCESS, 1167 = NOT_CONNECTED


def _xinput_occupied_slots() -> set:
    """Return the set of XInput slots (0–3) that report CONNECTED."""
    out: set = set()
    if _xinput_dll is None:
        return out
    for i in range(4):
        if _xinput_connected(i):
            out.add(i)
    return out


# DOLPHIN GAMEPAD INPUT

class DolphinGamepadInput:
    """Thread-safe virtual Xbox 360 controller for Dolphin.

    A daemon *pump thread* reads the desired stick/button state and
    writes HID reports at ``PUMP_HZ`` (default 120).  This keeps
    Dolphin's XInput polling happy regardless of how slow or bursty
    the CV main loop is.

    A monotonic heartbeat on the right trigger cycles through byte
    values 1→2→3→4→1→… so no two consecutive HID reports are
    byte-identical (prevents XInput duplicate-report suppression).

    Stick mapping (public API):
      stick_x: 0.0 = full left,  0.5 = centre, 1.0 = full right
      stick_y: 0.0 = full down,  0.5 = centre, 1.0 = full up

    Internally converted to vgamepad's −1.0 … +1.0 range.
    """

    PUMP_HZ: int = 120           # target HID report frequency
    VERIFY_SEC: float = 5.0      # XInput connectivity check interval
    RECONNECT_PAUSE: float = 0.3 # pause during device recreation

    def __init__(self) -> None:
        self._gp = None
        self._vg = None
        self._vbus_ref = None
        self._connected: bool = False

        # Desired output state
        self._sx: float = 0.0     # left stick X  (−1…+1)
        self._sy: float = 0.0     # left stick Y  (−1…+1)
        self._want_a: bool = False
        self._want_b: bool = False

        self._a_hw: bool = False
        self._b_hw: bool = False

        # Pump thread bookkeeping
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._report_count: int = 0
        self._error_count: int = 0
        self._heartbeat: int = 0
        self._consecutive_errors: int = 0
        self._last_pump_ts: float = 0.0
        self._last_health_check: float = 0.0

        # Device-level tracking
        self._xinput_slot: int = -1    # detected XInput slot
        self._device_gen: int = 0      # incremented on each recreation
        self._effective_hz: float = 0.0

        # Recovery / zombie detection
        self._device_recover_attempts: int = 0
        self._device_recover_ts: float = 0.0
        self._zombie_zero_count: int = 0
        self._last_roundtrip_check: float = 0.0

    # Lifecycle

    def connect(self) -> bool:
        """Create the virtual gamepad and start the 120 Hz pump.

        Returns True on success.
        """
        try:
            import vgamepad as vg
            self._vg = vg
        except ImportError:
            print("[INPUT] ERROR: vgamepad not installed.  "
                  "Run:  pip install vgamepad")
            return False

        try:
            self._vbus_ref = vg.win.virtual_gamepad.VBUS
        except AttributeError:
            pass  # internal layout changed — non-fatal

        pre_slots = _xinput_occupied_slots()

        try:
            self._gp = vg.VX360Gamepad()
            self._connected = True
        except Exception as exc:
            print(f"[INPUT] ERROR creating virtual gamepad: {exc}")
            return False

        self._prime_device()

        time.sleep(0.15)
        post_slots = _xinput_occupied_slots()
        new_slots = post_slots - pre_slots
        if new_slots:
            self._xinput_slot = min(new_slots)
        elif post_slots:
            self._xinput_slot = min(post_slots)
        else:
            self._xinput_slot = 0  # best guess

        if self._xinput_slot >= 0:
            print(f"[INPUT] Device on XInput slot {self._xinput_slot}")
        if _xinput_dll is None:
            print("[INPUT] (XInput DLL not found — "
                  "device verification disabled)")

        # Start background pump
        self._last_pump_ts = time.perf_counter()
        self._running = True
        self._thread = threading.Thread(
            target=self._pump_loop, daemon=True, name="gp-pump")
        self._thread.start()

        print("[INPUT] Virtual Xbox 360 gamepad created (vgamepad/ViGEm)")
        print(f"[INPUT] Pump thread active @ {self.PUMP_HZ} Hz "
              f"(gen {self._device_gen})")
        return True

    def _prime_device(self) -> None:
        """Send varied reports right after device creation."""
        gp = self._gp
        if gp is None:
            return
        for i in range(20):
            gp.left_joystick_float(0.0, 0.0)
            gp.right_trigger(1 + (i & 3))  # byte values 1→4
            gp.update()
            time.sleep(0.008)
        gp.right_trigger(0)
        gp.update()

    def close(self) -> None:
        """Stop the pump thread and release the virtual gamepad.

        Safe to call multiple times.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._gp is not None:
            try:
                self._gp.left_joystick_float(0.0, 0.0)
                if self._a_hw:
                    self._gp.release_button(
                        self._vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
                    self._a_hw = False
                if self._b_hw:
                    self._gp.release_button(
                        self._vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
                    self._b_hw = False
                self._gp.right_trigger(0)
                self._gp.update()
            except Exception:
                pass
            try:
                del self._gp
            except Exception:
                pass

        self._gp = None
        self._connected = False
        self._vbus_ref = None

        rpt = self._report_count
        err = self._error_count
        print(f"[INPUT] Gamepad closed  "
              f"({rpt} reports, {err} errors, "
              f"gen {self._device_gen})")

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def is_alive(self) -> bool:
        """True if pump thread is running and recently active."""
        if self._thread is None or not self._thread.is_alive():
            return False
        return (time.perf_counter() - self._last_pump_ts) < 0.5

    @property
    def effective_hz(self) -> float:
        """Measured pump rate over the last 1-second window."""
        return self._effective_hz

    @property
    def device_generation(self) -> int:
        """How many times the device has been (re)created."""
        return self._device_gen

    @property
    def xinput_slot(self) -> int:
        """Which XInput slot (0–3) our device occupies, or −1."""
        return self._xinput_slot

    def _ensure_pump_alive(self) -> None:
        """Check pump thread health; auto-restart if dead or stalled.

        Called from apply() — throttled to at most once per second.
        Also handles device-level recovery: if ``_gp`` is ``None``
        (device recreation failed earlier), attempts to create a new
        device before restarting the pump thread.
        """
        if not self._connected:
            return
        now = time.perf_counter()
        if now - self._last_health_check < 1.0:
            return
        self._last_health_check = now

        # Device-level recovery
        if self._gp is None:
            # Exponential backoff: 0.5s, 1s, 2s, 4s … capped at 10s
            backoff = min(10.0, 0.5 * (2 ** self._device_recover_attempts))
            if now - self._device_recover_ts < backoff:
                return
            self._device_recover_ts = now
            self._device_recover_attempts += 1
            print(f"[GAMEPAD] Device is None — recovery attempt "
                  f"{self._device_recover_attempts} "
                  f"(backoff {backoff:.1f}s)")
            try:
                new_gp = self._vg.VX360Gamepad()
                self._gp = new_gp
                self._prime_device()
                self._a_hw = False
                self._b_hw = False
                self._consecutive_errors = 0
                self._device_gen += 1
                self._device_recover_attempts = 0
                time.sleep(0.15)
                for idx in range(4):
                    if _xinput_connected(idx):
                        self._xinput_slot = idx
                        break
                print(f"[GAMEPAD] Device recovered — gen {self._device_gen}, "
                      f"xinput slot {self._xinput_slot}")
            except Exception as exc:
                print(f"[GAMEPAD] Device recovery failed: {exc}")
                self._gp = None
                return  # will retry on next tick after backoff

        # Zombie detection
        if (self._xinput_slot >= 0 and _xinput_dll is not None
                and now - self._last_roundtrip_check > self.VERIFY_SEC):
            self._last_roundtrip_check = now
            state = _XINPUT_STATE()
            rc = _xinput_dll.XInputGetState(
                self._xinput_slot, ctypes.byref(state))
            if rc != 0:
                # Slot disconnected
                print(f"[GAMEPAD] XInput slot {self._xinput_slot} "
                      f"DISCONNECTED (rc={rc}) — recreating")
                self._force_full_reconnect()
                return
            rt = state.Gamepad.bRightTrigger
            if rt == 0:
                self._zombie_zero_count += 1
            else:
                self._zombie_zero_count = 0
            if self._zombie_zero_count >= 3:
                print(f"[GAMEPAD] Zombie device detected — "
                      f"right trigger reads 0 for {self._zombie_zero_count} "
                      f"consecutive checks. Recreating.")
                self._zombie_zero_count = 0
                self._force_full_reconnect()
                return

        # Pump thread health
        needs_restart = False
        if self._thread is None or not self._thread.is_alive():
            print("[GAMEPAD] WARNING: pump thread died — restarting")
            needs_restart = True
        elif (now - self._last_pump_ts) > 0.5:
            elapsed = now - self._last_pump_ts
            print(f"[GAMEPAD] WARNING: pump stalled ({elapsed:.2f}s) "
                  "— restarting")
            needs_restart = True

        if needs_restart:
            self._running = False
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=1.0)
            time.sleep(0.05)
            self._running = True
            self._consecutive_errors = 0
            self._last_pump_ts = time.perf_counter()
            self._thread = threading.Thread(
                target=self._pump_loop, daemon=True, name="gp-pump")
            self._thread.start()
            print("[GAMEPAD] Pump thread restarted")

    def _force_full_reconnect(self) -> None:
        """Stop pump thread, destroy device, create new one, restart pump.

        Called when zombie detection or XInput disconnect is found from
        the main thread (via _ensure_pump_alive).
        """
        print("[GAMEPAD] >>> Full reconnect starting <<<")
        # Stop pump
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        # Destroy old device
        old_gp = self._gp
        self._gp = None
        if old_gp is not None:
            try:
                old_gp.left_joystick_float(0.0, 0.0)
                old_gp.right_trigger(0)
                old_gp.update()
            except Exception:
                pass
            try:
                del old_gp
            except Exception:
                pass
        time.sleep(self.RECONNECT_PAUSE)

        # Create new device
        try:
            new_gp = self._vg.VX360Gamepad()
            self._gp = new_gp
            self._prime_device()
            self._a_hw = False
            self._b_hw = False
            self._consecutive_errors = 0
            self._device_gen += 1
            self._device_recover_attempts = 0
            time.sleep(0.15)
            for idx in range(4):
                if _xinput_connected(idx):
                    self._xinput_slot = idx
                    break
            print(f"[GAMEPAD] Full reconnect OK — gen {self._device_gen}, "
                  f"xinput slot {self._xinput_slot}")
        except Exception as exc:
            print(f"[GAMEPAD] Full reconnect FAILED: {exc}")
            self._gp = None
            return

        # Restart pump
        self._running = True
        self._last_pump_ts = time.perf_counter()
        self._thread = threading.Thread(
            target=self._pump_loop, daemon=True, name="gp-pump")
        self._thread.start()
        print("[GAMEPAD] Pump thread restarted after full reconnect")

    # Pump thread

    def _pump_loop(self) -> None:
        """Background loop: writes HID reports at PUMP_HZ."""
        _begin_hires_timer()

        dt = 1.0 / self.PUMP_HZ
        alive_interval = self.PUMP_HZ * 15          # log every ~15 s
        verify_interval = max(1, int(self.VERIFY_SEC * self.PUMP_HZ))
        verify_countdown = verify_interval

        gp = self._gp
        vg = self._vg

        if gp is None or vg is None:
            print("[GAMEPAD] pump_loop: no device — exiting")
            _end_hires_timer()
            return

        btn_a = vg.XUSB_BUTTON.XUSB_GAMEPAD_A
        btn_b = vg.XUSB_BUTTON.XUSB_GAMEPAD_B

        # Hz measurement window
        hz_window_start = time.perf_counter()
        hz_window_count = 0

        # Precise timing
        next_tick = time.perf_counter()
        exit_reason = "unknown"

        try:
            while self._running:
                sx = self._sx
                sy = self._sy
                want_a = self._want_a
                want_b = self._want_b

                try:
                    gp.left_joystick_float(float(sx), float(sy))

                    # Button A
                    if want_a and not self._a_hw:
                        gp.press_button(btn_a)
                        self._a_hw = True
                    elif not want_a and self._a_hw:
                        gp.release_button(btn_a)
                        self._a_hw = False

                    # Button B
                    if want_b and not self._b_hw:
                        gp.press_button(btn_b)
                        self._b_hw = True
                    elif not want_b and self._b_hw:
                        gp.release_button(btn_b)
                        self._b_hw = False

                    self._heartbeat = (self._heartbeat % 4) + 1
                    gp.right_trigger(self._heartbeat)

                    gp.update()
                    self._report_count += 1
                    self._consecutive_errors = 0
                    self._last_pump_ts = time.perf_counter()

                    hz_window_count += 1
                    hz_elapsed = self._last_pump_ts - hz_window_start
                    if hz_elapsed >= 1.0:
                        self._effective_hz = hz_window_count / hz_elapsed
                        hz_window_start = self._last_pump_ts
                        hz_window_count = 0

                    # Periodic alive log
                    if (alive_interval
                            and self._report_count % alive_interval == 0):
                        print(
                            f"[GAMEPAD] alive — "
                            f"{self._report_count} reports  "
                            f"{self._effective_hz:.0f} Hz  "
                            f"stick=({sx:+.2f},{sy:+.2f})  "
                            f"A={'ON' if self._a_hw else 'off'}  "
                            f"B={'ON' if self._b_hw else 'off'}  "
                            f"xinput={self._xinput_slot}  "
                            f"gen={self._device_gen}")

                except Exception as exc:
                    self._error_count += 1
                    self._consecutive_errors += 1
                    if self._consecutive_errors <= 3:
                        print(f"[GAMEPAD] pump error "
                              f"#{self._consecutive_errors}: {exc}")

                    if self._consecutive_errors >= 50:
                        print("[GAMEPAD] 50 consecutive errors "
                              "— recreating device")
                        self._recreate_device()
                        gp = self._gp  # update local ref
                        if gp is None:
                            exit_reason = "device recreation failed"
                            break

                # Periodic XInput verification
                verify_countdown -= 1
                if verify_countdown <= 0:
                    verify_countdown = verify_interval
                    try:
                        if (self._xinput_slot >= 0
                                and _xinput_dll is not None):
                            if not _xinput_connected(self._xinput_slot):
                                print(
                                    f"[GAMEPAD] XInput slot "
                                    f"{self._xinput_slot} says "
                                    f"DISCONNECTED — recreating device")
                                self._recreate_device()
                                gp = self._gp
                                if gp is None:
                                    exit_reason = "XInput recovery failed"
                                    break
                    except Exception as _xi_exc:
                        print(f"[GAMEPAD] XInput check error: {_xi_exc}")

                # Precise timing
                next_tick += dt
                now = time.perf_counter()
                remaining = next_tick - now
                if remaining > 0.002:
                    time.sleep(remaining - 0.001)
                while time.perf_counter() < next_tick:
                    pass
                # Anti-drift: if we fell behind by > 2 ticks, reset
                if time.perf_counter() - next_tick > dt * 2:
                    next_tick = time.perf_counter()

            exit_reason = "clean (_running=False)"

        except Exception as exc:
            exit_reason = f"unhandled: {exc}"
            print(f"[GAMEPAD] FATAL pump error: {exc}")

        finally:
            _end_hires_timer()
            print(f"[GAMEPAD] Pump thread exited — {exit_reason}  "
                  f"({self._report_count} reports total, "
                  f"gen {self._device_gen})")

    def _recreate_device(self) -> None:
        """Destroy and recreate the ViGEm virtual device.

        Called from the pump thread when persistent errors or an
        XInput disconnect are detected.
        """
        self._last_pump_ts = time.perf_counter()

        old_gp = self._gp
        self._gp = None
        if old_gp is not None:
            try:
                old_gp.left_joystick_float(0.0, 0.0)
                old_gp.right_trigger(0)
                old_gp.update()
            except Exception:
                pass
            try:
                del old_gp
            except Exception:
                pass

        time.sleep(self.RECONNECT_PAUSE)

        try:
            new_gp = self._vg.VX360Gamepad()
            self._gp = new_gp
            self._prime_device()
            self._a_hw = False
            self._b_hw = False
            self._consecutive_errors = 0
            self._device_gen += 1
            self._device_recover_attempts = 0  # reset backoff
            self._zombie_zero_count = 0
            self._last_pump_ts = time.perf_counter()

            # Re-detect XInput slot
            time.sleep(0.15)
            for idx in range(4):
                if _xinput_connected(idx):
                    self._xinput_slot = idx
                    break

            print(f"[GAMEPAD] Device recreated — "
                  f"gen {self._device_gen}, "
                  f"xinput slot {self._xinput_slot}")
            print(f"[GAMEPAD] *** If Dolphin stopped responding, "
                  f"check Device = XInput/"
                  f"{self._xinput_slot}/Gamepad ***")

        except Exception as e2:
            print(f"[GAMEPAD] Recreation FAILED: {e2}")
            self._gp = None
            time.sleep(1.0)

    # Main-thread API

    def apply(self, stick_x: float, stick_y: float,
              swing: bool, press_b: bool = False) -> None:
        """Set desired gamepad state.  Non-blocking, thread-safe.

        The pump thread picks up the new values within ~8 ms.
        Includes a periodic health check that auto-restarts the
        pump thread if it has died or stalled.

        Parameters
        ----------
        stick_x : float   0.0 = full left, 0.5 = centre, 1.0 = full right
        stick_y : float   0.0 = full down, 0.5 = centre, 1.0 = full up
        swing   : bool    True → press A, False → release A
        press_b : bool    True → press B, False → release B
        """
        self._ensure_pump_alive()
        self._sx = (max(0.0, min(1.0, float(stick_x))) - 0.5) * 2.0
        self._sy = (max(0.0, min(1.0, float(stick_y))) - 0.5) * 2.0
        self._want_a = bool(swing)
        self._want_b = bool(press_b)

    def set_stick(self, x: float = 0.5, y: float = 0.5) -> None:
        """Set left stick position (0.0–1.0 range)."""
        self._sx = (max(0.0, min(1.0, x)) - 0.5) * 2.0
        self._sy = (max(0.0, min(1.0, y)) - 0.5) * 2.0

    def center_stick(self) -> None:
        """Return the stick to neutral."""
        self._sx = 0.0
        self._sy = 0.0

    def press_a(self) -> None:
        """Request A press (picked up by pump thread)."""
        self._want_a = True

    def release_a(self) -> None:
        """Request A release (picked up by pump thread)."""
        self._want_a = False

    def safe_neutral(self) -> None:
        """Emergency neutral: centre stick + release all buttons."""
        self._sx = 0.0
        self._sy = 0.0
        self._want_a = False
        self._want_b = False

    def press_b(self) -> None:
        """Request B press (picked up by pump thread)."""
        self._want_b = True

    def release_b(self) -> None:
        """Request B release (picked up by pump thread)."""
        self._want_b = False

    def tap_a(self, hold_sec: float = 0.05) -> None:
        """Press A, wait, release.  Blocking — only for testing."""
        self.press_a()
        time.sleep(hold_sec)
        self.release_a()

    def get_health(self) -> dict:
        """Return a diagnostic dict for the main loop HUD."""
        return {
            "alive": self.is_alive,
            "hz": self._effective_hz,
            "reports": self._report_count,
            "errors": self._error_count,
            "consec_err": self._consecutive_errors,
            "slot": self._xinput_slot,
            "gen": self._device_gen,
        }

    # Direct button helpers

    def press_button(self, btn_name: str) -> None:
        """Generic button press by name (A, B, X, Y, Start, Z)."""
        if self._gp is None or self._vg is None:
            return
        btn_map = {
            "A": self._vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
            "B": self._vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
            "X": self._vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
            "Y": self._vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
            "Start": self._vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
            "Z": self._vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
        }
        btn = btn_map.get(btn_name)
        if btn is not None:
            self._gp.press_button(btn)
            self._gp.update()

    def release_button(self, btn_name: str) -> None:
        """Generic button release by name."""
        if self._gp is None or self._vg is None:
            return
        btn_map = {
            "A": self._vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
            "B": self._vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
            "X": self._vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
            "Y": self._vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
            "Start": self._vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
            "Z": self._vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
        }
        btn = btn_map.get(btn_name)
        if btn is not None:
            self._gp.release_button(btn)
            self._gp.update()
