from __future__ import annotations

from typing import Protocol

from compneurovis.messages import CommandPayload, UpdatePayload


class Transport(Protocol):
    def start(self) -> None:
        pass

    def send(self, command: CommandPayload) -> None:
        pass

    def poll(self) -> list[UpdatePayload]:
        pass

    def stop(self) -> None:
        pass
