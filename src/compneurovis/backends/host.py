from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from compneurovis.backends.base import BackendBase
from compneurovis.core.actor import ActorSource
from compneurovis.core.app import AppSpec
from compneurovis.core.hosts import ActorHost, resolve_actor_source
from compneurovis.core.messages import AppSpecSnapshot, StopBackend
from compneurovis.transports import TransportEndpoint

if TYPE_CHECKING:
    from compneurovis.core.runtime import AppRuntime


class BackendHost(ActorHost):
    """ActorHost for backend actors. Handles StopBackend and drives BackendBase."""

    def __init__(self, endpoint: TransportEndpoint | None = None) -> None:
        super().__init__(endpoint=endpoint)
        self._stop_requested = False

    def start(self, actor_source: ActorSource, app_spec: AppSpec) -> BackendBase:
        actor = super().start(actor_source, app_spec)
        if not isinstance(actor, BackendBase):
            raise TypeError(f"BackendHost expected BackendBase, got {type(actor)!r}")
        # Backend is authoritative for the AppSpec. Announce it so a frontend
        # that started without one (multiprocess desktop path — no longer built
        # twice) can adopt it. ThreadBackendHost overrides start() and does not
        # announce: notebook is in-process, single build, frontend already has it.
        if app_spec is not None:
            actor.emit_update(AppSpecSnapshot(app_spec))
        return actor

    def receive(self) -> None:
        actor = self._actor()
        if self.endpoint is None:
            return
        for message in self.endpoint.poll():
            if isinstance(message.payload, StopBackend):
                self._stop_requested = True
                return
            actor.handle(message)

    def step(self) -> None:
        self.receive()
        if self.should_stop():
            return
        actor = self._actor()
        if isinstance(actor, BackendBase) and actor.is_live():
            actor.update()
        self.flush()

    def idle_sleep(self) -> float:
        actor = self._actor()
        if isinstance(actor, BackendBase):
            return actor.idle_sleep()
        return 1.0 / 60.0

    def should_stop(self) -> bool:
        return self._stop_requested


class ThreadBackendHost(BackendHost):
    """BackendHost whose step loop runs in a daemon thread.

    Use when the backend must share the same process as the frontend (e.g.
    .ipynb notebooks where no script file is available for ScriptBackendProcess).
    NEURON and JAX release the GIL during simulation, so real parallelism
    is achieved despite the thread model.
    """

    def __init__(
        self,
        actor_source: ActorSource,
        runtime: AppRuntime,
        endpoint: TransportEndpoint,
    ) -> None:
        super().__init__(endpoint=endpoint)
        self._actor_source = actor_source
        self._runtime = runtime
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.actor = resolve_actor_source(self._actor_source)
        self.actor.initialize(self._runtime.app_spec)
        if not isinstance(self.actor, BackendBase):
            raise TypeError(f"ThreadBackendHost expected BackendBase, got {type(self.actor)!r}")

    def run(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        try:
            while not self.should_stop():
                started = time.monotonic()
                self.step()
                remaining = self.idle_sleep() - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
        except (BrokenPipeError, OSError):
            pass
        finally:
            self.stop()

    def step(self) -> None:
        self.receive()
        if self.should_stop():
            return
        actor = self._actor()
        if isinstance(actor, BackendBase) and actor.is_live():
            t0 = time.monotonic()
            actor.update()
            dt_ms = (time.monotonic() - t0) * 1000
            _step_count = getattr(self, "_perf_step_count", 0) + 1
            self._perf_step_count = _step_count
            if _step_count % 60 == 0:
                _PERF_LOG = r"c:\Users\orren\Documents\PythonProjects\CompNeuroVis\scratch\perf_stats.txt"
                try:
                    with open(_PERF_LOG, "a") as _f:
                        _f.write(f"[backend_step] n={_step_count} update_ms={dt_ms:.2f} display_dt={getattr(actor, 'display_dt', '?')}\n")
                except Exception:
                    pass
        self.flush()


__all__ = ["BackendHost", "ThreadBackendHost"]
