"""
test_gamepad.py — Test virtual gamepad to verify Dolphin receives input.

Run this WHILE Dolphin is open with the game loaded.
It will cycle through stick movements and press A.

You should see the character move in-game if Dolphin is configured correctly.

Usage:
    python test_gamepad.py              # quick 15-second test
    python test_gamepad.py --long       # 5-minute endurance test
    python test_gamepad.py --long 600   # 10-minute endurance test
"""

import argparse
import ctypes
import math
import time
import sys

try:
    import vgamepad as vg
except ImportError:
    print("ERROR: pip install vgamepad")
    sys.exit(1)


# XInput verification

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
        ("dwPacketNumber", ctypes.c_ulong),
        ("Gamepad",        _XINPUT_GAMEPAD),
    ]


def _check_xinput():
    """Print which XInput slots are connected."""
    if _xinput_dll is None:
        print("  (XInput DLL not available — cannot verify)")
        return
    for i in range(4):
        state = _XINPUT_STATE()
        rc = _xinput_dll.XInputGetState(i, ctypes.byref(state))
        status = "CONNECTED" if rc == 0 else f"not connected (rc={rc})"
        print(f"  XInput slot {i}: {status}")


def quick_test():
    """Original quick test: directions + A presses (~15 seconds)."""
    print("Creating virtual Xbox 360 gamepad...")
    gp = vg.VX360Gamepad()
    print("Gamepad created!  ViGEm device active.")
    print()
    print("XInput slot status (before test):")
    _check_xinput()
    print()
    print("Switch to Dolphin NOW.  Tests start in 3 seconds...")
    print("(Make sure Dolphin Port 1 = XInput/0/Gamepad)")
    print()
    time.sleep(3)

    print("=" * 50)
    print("TEST 1: Move stick LEFT for 1 second")
    print("=" * 50)
    gp.left_joystick_float(-1.0, 0.0)
    gp.update()
    time.sleep(1.0)
    gp.left_joystick_float(0.0, 0.0)
    gp.update()
    time.sleep(0.5)

    print("TEST 2: Move stick RIGHT for 1 second")
    print("=" * 50)
    gp.left_joystick_float(1.0, 0.0)
    gp.update()
    time.sleep(1.0)
    gp.left_joystick_float(0.0, 0.0)
    gp.update()
    time.sleep(0.5)

    print("TEST 3: Move stick UP for 1 second")
    print("=" * 50)
    gp.left_joystick_float(0.0, 1.0)
    gp.update()
    time.sleep(1.0)
    gp.left_joystick_float(0.0, 0.0)
    gp.update()
    time.sleep(0.5)

    print("TEST 4: Move stick DOWN for 1 second")
    print("=" * 50)
    gp.left_joystick_float(0.0, -1.0)
    gp.update()
    time.sleep(1.0)
    gp.left_joystick_float(0.0, 0.0)
    gp.update()
    time.sleep(0.5)

    print("TEST 5: Press A button 3 times")
    print("=" * 50)
    for i in range(3):
        print(f"  Press A #{i+1}")
        gp.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        gp.update()
        time.sleep(0.15)
        gp.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        gp.update()
        time.sleep(0.5)

    print("TEST 6: Move stick in circle for 3 seconds")
    print("=" * 50)
    t0 = time.time()
    while time.time() - t0 < 3.0:
        angle = (time.time() - t0) * 2.0 * math.pi  # 1 rev/sec
        x = math.cos(angle)
        y = math.sin(angle)
        gp.left_joystick_float(x, y)
        gp.update()
        time.sleep(0.016)

    gp.left_joystick_float(0.0, 0.0)
    gp.update()

    print()
    print("=" * 50)
    print("DONE! Did the character move in Dolphin?")
    print()
    print("XInput slot status (after test):")
    _check_xinput()
    print()
    print("If YES: The gamepad works. Run bat_live.py")
    print("If NO:  Check Dolphin controller config:")
    print("  1. Controllers → Port 1 → Standard Controller → Configure")
    print("  2. Device = XInput/0/Gamepad")
    print("  3. Click 'Default' to auto-map all buttons")
    print("     (this maps Left Stick → Main Stick, A → A, etc.)")
    print("  4. Save the profile")
    print("  5. IMPORTANT: Enable 'Background Input' if available")
    print("=" * 50)

    del gp


def long_run_test(duration_sec: int = 300):
    """Endurance test using DolphinGamepadInput (threaded pump).

    Runs for ``duration_sec`` seconds, sending gentle stick circles
    and periodic A presses.  Proves the gamepad stays alive indefinitely.

    The main loop INTENTIONALLY runs at only ~15 Hz (simulating a slow
    CV loop) while the background pump thread keeps the HID reports
    flowing at ~120 Hz.

    Includes round-trip XInput verification: reads back the right-trigger
    heartbeat value to verify reports actually reach XInput.

    Pass/fail criteria:
    - PASS: character responds the ENTIRE duration, no disconnects.
    - FAIL: character stops responding at any point.
    - Check: "XInput slot N: CONNECTED" at regular intervals.
    - Check: pump Hz stays at ~110-120, never drops to 0.
    - Check: right trigger readback is never 0 for 3+ consecutive checks.
    """
    # Import the threaded gamepad class
    sys.path.insert(0, ".")
    from msb.dolphin_input import DolphinGamepadInput

    print()
    print("XInput slots BEFORE device creation:")
    _check_xinput()
    print()

    inp = DolphinGamepadInput()
    if not inp.connect():
        print("Failed to connect gamepad")
        return

    print()
    print("=" * 64)
    print(f"  LONG-RUN ENDURANCE TEST  ({duration_sec} seconds)")
    print("=" * 64)
    print("  The character should move in gentle circles and")
    print("  press A every 10 seconds.  If the character stops")
    print("  responding at ANY point, the gamepad keepalive is broken.")
    print()
    print("  XInput verification runs every 10 seconds.")
    print("  Pump Hz is reported every 10 seconds.")
    print("  Round-trip heartbeat check included.")
    print()
    print("  Switch to Dolphin NOW.  Test starts in 3 seconds...")
    print("=" * 64)
    time.sleep(3)

    t0 = time.time()
    cycle = 0
    last_log = t0
    xinput_disconnects = 0
    hz_measurements = []
    zombie_warnings = 0

    try:
        while True:
            elapsed = time.time() - t0
            if elapsed >= duration_sec:
                break

            # Gentle circle (one revolution every 5 seconds)
            angle = (elapsed / 5.0) * 2.0 * math.pi
            sx = 0.5 + 0.35 * math.cos(angle)
            sy = 0.5 + 0.35 * math.sin(angle)

            # Press A for 0.5s every 10 seconds
            swing = (int(elapsed) % 10) < 1

            inp.apply(sx, sy, swing)
            cycle += 1

            # Status log + XInput verification every 10 seconds
            if time.time() - last_log >= 10.0:
                last_log = time.time()
                hp = inp.get_health()
                hz_measurements.append(hp['hz'])

                # XInput slot check
                xi_slot = inp.xinput_slot
                xi_ok = True
                rt_val = -1
                if _xinput_dll is not None and xi_slot >= 0:
                    state = _XINPUT_STATE()
                    rc = _xinput_dll.XInputGetState(
                        xi_slot, ctypes.byref(state))
                    xi_ok = (rc == 0)
                    if not xi_ok:
                        xinput_disconnects += 1
                    else:
                        rt_val = state.Gamepad.bRightTrigger
                        if rt_val == 0:
                            zombie_warnings += 1

                xi_str = ("CONNECTED" if xi_ok
                          else "*** DISCONNECTED ***")
                rt_str = (f"rt={rt_val}" if rt_val >= 0 else "rt=?")

                print(
                    f"  [{elapsed:.0f}s / {duration_sec}s]  "
                    f"cycles={cycle}  "
                    f"pump={hp['hz']:.0f}Hz  "
                    f"reports={hp['reports']}  "
                    f"errors={hp['errors']}  "
                    f"gen={hp['gen']}  "
                    f"stick=({sx:.2f},{sy:.2f})  "
                    f"A={'PRESS' if swing else 'off'}  "
                    f"XInput[{xi_slot}]={xi_str}  "
                    f"{rt_str}")

            time.sleep(0.066)

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")

    elapsed = time.time() - t0
    avg_hz = (sum(hz_measurements) / len(hz_measurements)
              if hz_measurements else 0.0)
    min_hz = min(hz_measurements) if hz_measurements else 0.0
    max_hz = max(hz_measurements) if hz_measurements else 0.0

    print()
    print("=" * 64)
    print(f"  RESULTS")
    print("=" * 64)
    print(f"  Duration:       {elapsed:.0f}s")
    print(f"  Main loops:     {cycle}")
    print(f"  HID reports:    {inp._report_count}")
    print(f"  Pump errors:    {inp._error_count}")
    print(f"  Device gens:    {inp._device_gen}")
    print(f"  Pump Hz:        avg={avg_hz:.0f}  "
          f"min={min_hz:.0f}  max={max_hz:.0f}")
    print(f"  XInput disconnects detected: {xinput_disconnects}")
    print(f"  Zombie warnings (rt=0):      {zombie_warnings}")
    print()
    fail = False
    if xinput_disconnects > 0:
        print("  *** FAIL: XInput device disconnected during test ***")
        print("  The virtual device disappeared from Windows.")
        print("  Check ViGEmBus driver status and Windows power settings.")
        fail = True
    if zombie_warnings >= 3:
        print("  *** FAIL: Zombie device detected (rt=0 repeatedly) ***")
        print("  Reports are being sent but not reaching XInput.")
        fail = True
    if avg_hz < 50:
        print("  *** WARN: Pump Hz very low — timer resolution issue ***")
    if not fail:
        print("  If the character responded the ENTIRE time → PASS")
        print("  If it stopped at some point → FAIL (input issue)")
    print()
    print("  XInput slots (final):")
    _check_xinput()
    print("=" * 64)

    inp.safe_neutral()
    time.sleep(0.3)
    inp.close()


def main():
    ap = argparse.ArgumentParser(
        description="Test virtual gamepad for Dolphin")
    ap.add_argument("--long", nargs="?", const=300, type=int,
                    metavar="SECONDS",
                    help="Long-run endurance test (default: 300s)")
    args = ap.parse_args()

    if args.long is not None:
        long_run_test(args.long)
    else:
        quick_test()


if __name__ == "__main__":
    main()
