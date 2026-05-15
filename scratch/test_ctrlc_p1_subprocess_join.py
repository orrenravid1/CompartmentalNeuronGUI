"""
Test P1: does process.join(timeout=1) hang after subprocess gets KeyboardInterrupt?

Run: python scratch/test_ctrlc_p1_subprocess_join.py
Then Ctrl+C.
Expected if P1 is bug: hangs after Ctrl+C (join never returns).
Expected if NOT: prints "join returned" quickly.
"""
import multiprocessing as mp
import time


def worker():
    try:
        while True:
            time.sleep(0.016)
    except KeyboardInterrupt:
        print("[subprocess] KeyboardInterrupt caught, exiting")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    stop = mp.Event()
    p = mp.Process(target=worker)
    p.start()
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt, joining subprocess...")
        stop.set()
        p.join(timeout=2)
        print(f"[main] join returned, alive={p.is_alive()}")
        if p.is_alive():
            p.terminate()
