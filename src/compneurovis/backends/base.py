from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, TypeAlias

from compneurovis.actors import MessageActor
from compneurovis.core.app import AppSpec
from compneurovis.messages import CommandMessage, UpdateMessage, UpdatePayload, update_message


class Backend(MessageActor[CommandMessage, UpdateMessage], ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def initialize(self, app_spec: AppSpec) -> None:
        pass

    @abstractmethod
    def advance(self) -> None:
        pass

    @abstractmethod
    def handle(self, message: CommandMessage) -> None:
        pass

    def emit_update(self, update: UpdatePayload) -> None:
        self.emit(update_message(update))

    def is_live(self) -> bool:
        return True

    def idle_sleep(self) -> float:
        return 0.05

    def shutdown(self) -> None:
        pass


class BufferedBackend(Backend):
    pass


BackendFactory: TypeAlias = Callable[[], Backend]
BackendSource: TypeAlias = type[Backend] | BackendFactory


def resolve_backend_source(source: BackendSource) -> Backend:
    if isinstance(source, type):
        if not issubclass(source, Backend):
            raise TypeError(f"Expected Backend subclass, got {source!r}")
        return source()
    if isinstance(source, Backend):
        raise TypeError(
            "Eager backend instances are not supported for worker-backed apps. "
            "Pass a Backend subclass or a top-level zero-argument factory instead."
        )
    if callable(source):
        backend = source()
        if not isinstance(backend, Backend):
            raise TypeError(f"Backend factory returned {type(backend)!r}, expected Backend")
        return backend
    raise TypeError(f"Unsupported backend source: {source!r}")


def resolve_interaction_target_source(source: Any | None) -> Any | None:
    if source is None:
        return None
    if isinstance(source, type):
        return source()
    if callable(source) and not any(hasattr(source, attr) for attr in ("on_action", "on_key_press", "on_entity_clicked")):
        return source()
    return source
