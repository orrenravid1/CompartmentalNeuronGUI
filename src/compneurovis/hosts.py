from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypeAlias

from compneurovis._perf import clear_perf_logging_configuration, configure_perf_logging, perf_log
from compneurovis.core.actor import ActorBase, ActorSource
from compneurovis.core.app import AppSpec, DiagnosticsSpec
from compneurovis.messages import Error, update_message
from compneurovis.transports import TransportEndpoint


class Startable(Protocol):
    """Uniform lifecycle interface for in-process hosts and subprocess actors."""
    def start(self) -> None: ...
    def run(self) -> None: ...
    def stop(self) -> None: ...


# Callable[[AppSpec, TransportEndpoint | None], Startable]
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


__all__ = [
    "ActorHost",
    "ActorHostSource",
    "ActorProcess",
    "Startable",
    "TransportFactory",
    "configure_diagnostics",
    "configure_multiprocessing",
    "resolve_actor_source",
    "resolve_interaction_target_source",
]
