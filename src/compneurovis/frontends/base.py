from __future__ import annotations

from compneurovis.actors import MessageActor
from compneurovis.core.app import AppSpec
from compneurovis.messages import CommandPayload, Message, MessagePayload, command_message


class FrontendBase(MessageActor[Message[MessagePayload], Message[MessagePayload]]):
    def initialize(self, app_spec: AppSpec) -> None:
        pass

    def emit_command(self, command: CommandPayload) -> None:
        self.emit(command_message(command))

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
