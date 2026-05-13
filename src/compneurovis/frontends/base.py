from __future__ import annotations

from compneurovis.actors import MessageActor
from compneurovis.messages import CommandMessage, CommandPayload, UpdateMessage, command_message


class Frontend(MessageActor[UpdateMessage, CommandMessage]):
    def initialize(self) -> None:
        pass

    def emit_command(self, command: CommandPayload) -> None:
        self.emit(command_message(command))

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
