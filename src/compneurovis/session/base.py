from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Deque

from compneurovis.core.document import Document
from compneurovis.session.protocol import SessionCommand, SessionUpdate


class Session(ABC):
    @abstractmethod
    def initialize(self) -> Document | None:
        pass

    @abstractmethod
    def advance(self) -> None:
        pass

    @abstractmethod
    def handle(self, command: SessionCommand) -> None:
        pass

    @abstractmethod
    def read_updates(self) -> list[SessionUpdate]:
        pass

    def is_live(self) -> bool:
        return True

    def idle_sleep(self) -> float:
        return 0.05

    def shutdown(self) -> None:
        pass


class BufferedSession(Session):
    def __init__(self) -> None:
        self._updates: Deque[SessionUpdate] = deque()

    def emit(self, update: SessionUpdate) -> None:
        self._updates.append(update)

    def read_updates(self) -> list[SessionUpdate]:
        updates = list(self._updates)
        self._updates.clear()
        return updates

