"""
Test possibility 2: subprocess re-importing __main__ triggers use()/Qt warning.
Does importing frontend.py (with use() at module level) in a subprocess cause the warning?

Run: python scratch/test_p2_subprocess_use.py
Expected if P2 is the bug: "QWidget: Must construct..." warning printed from subprocess.
Expected if P2 is NOT the bug: subprocess prints "ok, no warning" cleanly.
"""
import multiprocessing as mp


def worker():
    # Simulate what subprocess does when it re-imports __main__ to unpickle SineBackend
    print("[subprocess] importing frontend.py (triggers use())")
    from compneurovis.frontends.vispy.frontend import VispyFrontendWindow  # noqa: F401
    print("[subprocess] done — no crash")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    p = mp.Process(target=worker)
    p.start()
    p.join()
    print(f"[main] subprocess exit code: {p.exitcode}")
