"""
Test P3: does runtime.wait() thread-join hang after foreground run() returns?

Run: python scratch/test_ctrlc_p3_wait_hang.py
Then Ctrl+C.
Expected if P3 is bug: hangs in thread join after Ctrl+C.
Expected if NOT: background thread exits cleanly within 1s.
"""
import threading
import time


def background():
    try:
        while True:
            time.sleep(0.016)
    except Exception:
        pass
    print("[bg] thread exiting")


def foreground():
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("[fg] KeyboardInterrupt caught")


t = threading.Thread(target=background, daemon=True)
t.start()

try:
    foreground()
except KeyboardInterrupt:
    pass

print("[main] joining background thread...")
t.join(timeout=2)
print(f"[main] join done, alive={t.is_alive()}")
