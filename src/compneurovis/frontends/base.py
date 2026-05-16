from __future__ import annotations

from compneurovis.core.actor import ActorBase, ActorRole


class FrontendBase(ActorBase):
    role = ActorRole.FRONTEND
