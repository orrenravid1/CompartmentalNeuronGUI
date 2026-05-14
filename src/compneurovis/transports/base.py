from __future__ import annotations

from typing import Protocol, TypeAlias, runtime_checkable

from compneurovis.messages import Message, MessagePayload


@runtime_checkable
class TransportEndpoint(Protocol):
    def send(self, message: Message[MessagePayload]) -> None: ...

    def poll(self) -> list[Message[MessagePayload]]: ...

    def close(self) -> None: ...


Transport: TypeAlias = TransportEndpoint

__all__ = ["Transport", "TransportEndpoint"]
