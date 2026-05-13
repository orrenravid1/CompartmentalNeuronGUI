from __future__ import annotations

from collections import deque
from typing import Any, Deque, Generic, TypeVar

from compneurovis.messages import Message

InboundMessageT = TypeVar("InboundMessageT", bound=Message[Any])
OutboundMessageT = TypeVar("OutboundMessageT", bound=Message[Any])


class MessageActor(Generic[InboundMessageT, OutboundMessageT]):
    def __init__(self) -> None:
        self._outbound_messages: Deque[OutboundMessageT] = deque()

    def handle(self, message: InboundMessageT) -> None:
        raise NotImplementedError

    def emit(self, message: OutboundMessageT) -> None:
        self._outbound_messages.append(message)

    def take_outbound_messages(self) -> list[OutboundMessageT]:
        messages = list(self._outbound_messages)
        self._outbound_messages.clear()
        return messages
