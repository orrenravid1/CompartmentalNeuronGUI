from __future__ import annotations

from collections import deque
from enum import Enum
from typing import TYPE_CHECKING, Callable, ClassVar, Deque, TypeAlias

# ActorBase is intentionally NOT ABC — FrontendBase must co-inherit with QMainWindow
# (PyQt6 sip.wrappertype) which conflicts with ABCMeta. BackendBase adds no ABC either,
# for symmetry.

if TYPE_CHECKING:
    from compneurovis.core.app import AppSpec
    from compneurovis.core.messages import Message, MessagePayload


class ActorRole(str, Enum):
    BACKEND = "backend"
    FRONTEND = "frontend"


class ActorBase:
    role: ClassVar[ActorRole]

    def __init__(self) -> None:
        self._outbound_messages: Deque[Message[MessagePayload]] = deque()

    def initialize(self, app_spec: AppSpec) -> None:
        pass

    def handle(self, message: Message[MessagePayload]) -> None:
        raise NotImplementedError

    def emit(self, message: Message[MessagePayload]) -> None:
        self._outbound_messages.append(message)

    def take_outbound_messages(self) -> list[Message[MessagePayload]]:
        messages = list(self._outbound_messages)
        self._outbound_messages.clear()
        return messages

    def shutdown(self) -> None:
        pass


ActorSource: TypeAlias = type[ActorBase] | Callable[[], ActorBase]
