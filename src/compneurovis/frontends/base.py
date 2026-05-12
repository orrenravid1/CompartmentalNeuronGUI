from __future__ import annotations

from typing import Protocol

from compneurovis.messages import CommandPayload, UpdatePayload


class Frontend(Protocol):
    def initialize(self) -> None:
        pass

    def handle(self, update: UpdatePayload) -> None:
        pass

    def take_outbound_messages(self) -> list[CommandPayload]:
        pass

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
