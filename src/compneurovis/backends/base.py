from __future__ import annotations

from compneurovis.core.actor import ActorBase, ActorRole
from compneurovis.messages import UpdatePayload, update_message


class BackendBase(ActorBase):
    role = ActorRole.BACKEND

    def advance(self) -> None:
        raise NotImplementedError

    def emit_update(self, update: UpdatePayload) -> None:
        self.emit(update_message(update))

    def is_live(self) -> bool:
        return True

    def idle_sleep(self) -> float:
        return 0.05
