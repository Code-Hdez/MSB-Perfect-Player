"""Move the mouse and left-click to print coordinates. Close with Ctrl+C."""
import time

try:
    import pynput.mouse as m
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pynput"])
    import pynput.mouse as m

print("Move the mouse to the TOP-LEFT corner of the game area and left-click...")
print("Then move it to the BOTTOM-RIGHT corner and left-click.")
print("Ctrl+C to exit.\n")

clicks = []

def on_click(x, y, button, pressed):
    if pressed and button == m.Button.left:
        label = ["top-left (left, top)", "bottom-right (right, bottom)"][len(clicks)]
        print(f"  Clic {len(clicks)+1} — {label}: ({x}, {y})")
        clicks.append((x, y))
        if len(clicks) == 2:
            l, t = clicks[0]
            r, b = clicks[1]
            print(f"\n>>> ROI = ({l}, {t}, {r}, {b})")
            print(f">>> Size: {r-l}x{b-t} px")
            return False  # stops the listener

with m.Listener(on_click=on_click) as listener:
    listener.join()
