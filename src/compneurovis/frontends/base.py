from __future__ import annotations

from compneurovis.core.actor import ActorBase, ActorRole
from compneurovis.messages import CommandPayload, command_message


class FrontendBase(ActorBase):
    role = ActorRole.FRONTEND

    def emit_command(self, command: CommandPayload) -> None:
        self.emit(command_message(command))
