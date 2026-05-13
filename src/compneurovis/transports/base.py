from __future__ import annotations

from typing import Protocol

from compneurovis.messages import CommandMessage, UpdateMessage


class Transport(Protocol):
    def start(self) -> None:
        pass

    def send(self, message: CommandMessage) -> None:
        pass

    def poll(self) -> list[UpdateMessage]:
        pass

    def stop(self) -> None:
        pass
