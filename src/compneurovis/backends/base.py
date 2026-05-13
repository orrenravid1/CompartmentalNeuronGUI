from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from compneurovis.actors import MessageActor
from compneurovis.core.app import AppSpec, BackendProtocol, BackendSource
from compneurovis.messages import Message, MessagePayload, UpdatePayload, update_message


class BackendBase(MessageActor[Message[MessagePayload], Message[MessagePayload]], ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def initialize(self, app_spec: AppSpec) -> None:
        pass

    @abstractmethod
    def advance(self) -> None:
        pass

    @abstractmethod
    def handle(self, message: Message[MessagePayload]) -> None:
        pass

    def emit_update(self, update: UpdatePayload) -> None:
        self.emit(update_message(update))

    def is_live(self) -> bool:
        return True

    def idle_sleep(self) -> float:
        return 0.05

    def shutdown(self) -> None:
        pass


def resolve_backend_source(source: BackendSource) -> BackendBase:
    if isinstance(source, type):
        if not issubclass(source, BackendBase):
            raise TypeError(f"Expected BackendBase subclass, got {source!r}")
        return source()
    if isinstance(source, BackendBase):
        raise TypeError(
            "Eager backend instances are not supported for worker-backed apps. "
            "Pass a BackendBase subclass or a top-level zero-argument factory instead."
        )
    if callable(source):
        backend = source()
        if not isinstance(backend, BackendBase):
            raise TypeError(f"Backend factory returned {type(backend)!r}, expected BackendBase")
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
