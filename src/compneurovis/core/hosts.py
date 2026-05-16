from __future__ import annotations

import multiprocessing as mp
import runpy
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypeAlias

from compneurovis.core._perf import clear_perf_logging_configuration, configure_perf_logging, perf_log
from compneurovis.core.actor import ActorBase, ActorSource
from compneurovis.core.app import AppSpec, DiagnosticsSpec
from compneurovis.core.messages import Error, update_message
from compneurovis.transports import TransportEndpoint


class Startable(Protocol):
    """Uniform lifecycle interface for in-process hosts and subprocess actors."""
    def start(self) -> None: ...
    def run(self) -> None: ...
    def stop(self) -> None: ...


# Callable[[AppRuntime, TransportEndpoint | None], Startable]
ActorHostSource: TypeAlias = Callable[..., Startable]
# Callable[[list[ActorSpec]], dict[str, TransportEndpoint]]
TransportFactory: TypeAlias = Callable[..., dict[str, TransportEndpoint]]


def configure_multiprocessing() -> None:
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)


def resolve_interaction_target_source(source: Any | None) -> Any | None:
    if source is None:
        return None
    if isinstance(source, type):
        return source()
    if callable(source) and not any(hasattr(source, attr) for attr in ("on_action", "on_key_press", "on_entity_clicked")):
        return source()
    return source


def resolve_actor_source(source: ActorSource) -> ActorBase:
    if isinstance(source, type):
        return source()
    if callable(source):
        return source()
    raise TypeError(f"Unsupported actor source: {source!r}")


def configure_diagnostics(diagnostics: DiagnosticsSpec | None) -> None:
    if diagnostics is None:
        clear_perf_logging_configuration()
    else:
        configure_perf_logging(diagnostics)


class ConnectionSlotHost:
    """Holds a transport endpoint open for a remotely-connected actor.

    Used by run_orchestrator for actors with host_source=None. Does not spawn
    or own any process — the remote actor connects independently.
    """

    def __init__(self, endpoint: TransportEndpoint | None = None) -> None:
        self.endpoint = endpoint

    def start(self) -> None:
        pass

    def run(self) -> None:
        pass

    def stop(self) -> None:
        if self.endpoint is not None:
            self.endpoint.close()


class ActorHost:
    def __init__(self, endpoint: TransportEndpoint | None = None) -> None:
        self.endpoint = endpoint
        self.actor: ActorBase | None = None

    def start(self, actor_source: ActorSource, app_spec: AppSpec) -> ActorBase:
        self.actor = resolve_actor_source(actor_source)
        self.actor.initialize(app_spec)
        return self.actor

    def receive(self) -> None:
        actor = self._actor()
        if self.endpoint is None:
            return
        for message in self.endpoint.poll():
            actor.handle(message)

    def flush(self) -> None:
        actor = self._actor()
        if self.endpoint is None:
            actor.take_outbound_messages()
            return
        for message in actor.take_outbound_messages():
            self.endpoint.send(message)

    def step(self) -> None:
        self.receive()
        self.flush()

    def idle_sleep(self) -> float:
        return 0.0

    def should_stop(self) -> bool:
        return False

    def stop(self) -> None:
        if self.actor is not None:
            self.actor.shutdown()
        if self.endpoint is not None:
            self.endpoint.close()

    def _actor(self) -> ActorBase:
        if self.actor is None:
            raise RuntimeError("ActorHost.start() must be called before stepping.")
        return self.actor


def _actor_process_worker(
    actor_source: ActorSource,
    app_spec: AppSpec,
    endpoint: TransportEndpoint,
    host_class: type[ActorHost],
    diagnostics: DiagnosticsSpec | None,
    stop_event,
) -> None:
    host = host_class(endpoint=endpoint)
    try:
        configure_diagnostics(diagnostics)
        host.start(actor_source, app_spec)
        perf_log("actor_process", "initialize", host_type=host_class.__name__)
        while not stop_event.is_set() and not host.should_stop():
            started = time.monotonic()
            host.step()
            delay = host.idle_sleep()
            if delay > 0:
                remaining = delay - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
    except Exception as exc:  # pragma: no cover - worker safety net
        detail = "".join(traceback.format_exception(exc))
        perf_log("actor_process", "error", error_type=type(exc).__name__, message=str(exc))
        endpoint.send(update_message(Error(detail)))
    finally:
        host.stop()


@dataclass(slots=True)
class ActorProcess:
    actor_source: ActorSource
    app_spec: AppSpec
    endpoint: TransportEndpoint
    host_class: type[ActorHost] = field(default=ActorHost)
    diagnostics: DiagnosticsSpec | None = None
    _stop_event: Any = field(init=False)
    _process: mp.Process | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._stop_event = mp.Event()

    def start(self) -> None:
        process = mp.Process(
            target=_actor_process_worker,
            args=(self.actor_source, self.app_spec, self.endpoint, self.host_class, self.diagnostics, self._stop_event),
        )
        process.start()
        self._process = process
        self.endpoint.close()

    def run(self) -> None:
        pass

    def stop(self) -> None:
        self._stop_event.set()
        if self._process is not None:
            self._process.join(timeout=1)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join()


# --------------------------------------------------------------------------- #
# Script-backend — subprocess launched by re-running the user's script        #
# --------------------------------------------------------------------------- #

_g_script_backend_endpoint: TransportEndpoint | None = None


def get_script_backend_endpoint() -> TransportEndpoint | None:
    """Return the endpoint if this process was spawned as a script backend."""
    return _g_script_backend_endpoint


def _set_script_backend_endpoint(ep: TransportEndpoint) -> None:
    global _g_script_backend_endpoint
    _g_script_backend_endpoint = ep


def _script_backend_worker(script_path: str, endpoint: TransportEndpoint) -> None:
    """Subprocess entry point for script-based backends.

    Sets the process-level endpoint flag then re-runs the user's script as
    __main__. The script detects get_script_backend_endpoint() is set and
    runs as a backend actor.

    Must be top-level in this module so multiprocessing can resolve it by
    qualified name. Do not move or rename.
    """
    _set_script_backend_endpoint(endpoint)
    inline_module = sys.modules.get("compneurovis.inline")
    reset_inline = getattr(inline_module, "_reset_inline_session", None)
    if callable(reset_inline):
        reset_inline()
    runpy.run_path(script_path, run_name="__main__")


class ScriptBackendProcess:
    """Startable that spawns a backend subprocess by re-running a script.

    The script is the backend. Pickling the backend state is not required —
    it is reconstructed fresh when the script re-runs. This is the right
    strategy for NEURON, JAX, and any model that builds non-picklable state.
    """

    def __init__(self, script_path: str, endpoint: TransportEndpoint) -> None:
        self._script_path = script_path
        self._endpoint = endpoint
        self._process: mp.Process | None = None

    def start(self) -> None:
        self._process = mp.Process(
            target=_script_backend_worker,
            args=(self._script_path, self._endpoint),
        )
        self._process.start()
        self._endpoint.close()

    def run(self) -> None:
        pass

    def stop(self) -> None:
        if self._process is None:
            return
        self._process.join(timeout=2)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join()


# --------------------------------------------------------------------------- #
# Thread-backend — for notebook contexts where subprocess is not available     #
# --------------------------------------------------------------------------- #

class ThreadBackendHost(ActorHost):
    """BackendHost whose step loop runs in a daemon thread.

    Use when the backend must share the same process as the frontend (e.g.
    .ipynb notebooks where no script file is available for ScriptBackendProcess).
    NEURON and JAX release the GIL during simulation, so real parallelism
    is achieved despite the thread model.
    """

    def __init__(
        self,
        actor_source: ActorSource,
        runtime: "AppRuntime",  # type: ignore[name-defined]  # avoid circular import
        endpoint: TransportEndpoint,
    ) -> None:
        super().__init__(endpoint=endpoint)
        self._actor_source = actor_source
        self._runtime = runtime
        self._thread: threading.Thread | None = None
        self._stop_requested = False

    def start(self) -> None:
        self.actor = resolve_actor_source(self._actor_source)
        self.actor.initialize(self._runtime.app_spec)

    def run(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def should_stop(self) -> bool:
        return self._stop_requested

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

    def idle_sleep(self) -> float:
        actor = self._actor()
        from compneurovis.backends.base import BackendBase
        if isinstance(actor, BackendBase):
            return actor.idle_sleep()
        return 1.0 / 60.0

    def step(self) -> None:
        self.receive()
        if self.should_stop():
            return
        actor = self._actor()
        from compneurovis.backends.base import BackendBase
        if isinstance(actor, BackendBase) and actor.is_live():
            actor.advance()
        self.flush()


# --------------------------------------------------------------------------- #
# AppHandle — returned by start_app() for non-blocking runs                   #
# --------------------------------------------------------------------------- #

class AppHandle:
    """Handle for a non-blocking app run (notebook / headless).

    Returned by start_app(). Call stop() to shut everything down.
    results maps actor_id → return value of host.run() (e.g. notebook widget).
    """

    def __init__(
        self,
        runtime: "AppRuntime",  # type: ignore[name-defined]
        items: list,
        results: dict,
    ) -> None:
        self._runtime = runtime
        self._items = items
        self.results = results

    def widget(self, actor_id: str = "frontend") -> Any:
        """Return the widget produced by the named actor's run()."""
        return self.results.get(actor_id)

    def wait(self) -> None:
        """Block until the foreground actor (e.g. Qt event loop) exits, then stop all actors.

        Call from the main thread. No-op if no foreground actor is present (notebook path).
        """
        fg = [(spec, host) for spec, host in self._items if spec.runs_in_foreground]
        if not fg:
            return
        _, fg_host = fg[0]
        try:
            fg_host.run()
        finally:
            self.stop()

    def stop(self) -> None:
        self._runtime.stop()
        for _, host in reversed(self._items):
            host.stop()


__all__ = [
    "ActorHost",
    "ActorHostSource",
    "ActorProcess",
    "AppHandle",
    "ConnectionSlotHost",
    "ScriptBackendProcess",
    "Startable",
    "ThreadBackendHost",
    "TransportFactory",
    "configure_diagnostics",
    "configure_multiprocessing",
    "get_script_backend_endpoint",
    "resolve_actor_source",
    "resolve_interaction_target_source",
]
