"""
Test possibility 3: pipe closes before subprocess can send — is the pipe lifetime
managed correctly when ActorProcess.start() closes the parent-side endpoint?

Run: python scratch/test_p3_pipe_lifetime.py
Expected if P3 is the bug: BrokenPipeError even with QApplication stable.
Expected if P3 is NOT the bug: messages received cleanly.
"""
import multiprocessing as mp
import time
from compneurovis.transports.pipe import make_pipe_pair


def sender(endpoint):
    """Simulate backend subprocess: send 3 messages then stop."""
    for i in range(3):
        time.sleep(0.1)
        print(f"[subprocess] sending msg {i}")
        try:
            endpoint.send(f"msg-{i}")  # type: ignore[arg-type]
            print(f"[subprocess] sent ok")
        except Exception as e:
            print(f"[subprocess] SEND FAILED: {e}")
    endpoint.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    pair = make_pipe_pair(left_name="backend", right_name="frontend")

    p = mp.Process(target=sender, args=(pair.left,))
    p.start()
    pair.left.close()  # close parent-side copy, same as ActorProcess.start() does

    # Simulate frontend polling
    time.sleep(0.5)
    msgs = pair.right._inbound.recv() if pair.right._inbound.poll() else None
    print(f"[main] received: {msgs}")
    pair.right.close()
    p.join()
    print(f"[main] subprocess exit code: {p.exitcode}")
