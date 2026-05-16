from __future__ import annotations

from compneurovis.core.actor import ActorBase, ActorRole


class BackendBase(ActorBase):
    role = ActorRole.BACKEND

    def update(self) -> None:
        raise NotImplementedError

    def is_live(self) -> bool:
        return True

    def idle_sleep(self) -> float:
        return 0.05
