from __future__ import annotations

from compneurovis.core.actor import ActorSource
from compneurovis.core.app import AppSpec
from compneurovis.frontends.base import FrontendBase
from compneurovis.hosts import ActorHost


class FrontendHost(ActorHost):
    def start(self, actor_source: ActorSource, app_spec: AppSpec) -> FrontendBase:
        actor = super().start(actor_source, app_spec)
        if not isinstance(actor, FrontendBase):
            raise TypeError(f"FrontendHost expected FrontendBase, got {type(actor)!r}")
        return actor

    def run(self) -> None:
        pass

    def _frontend(self) -> FrontendBase:
        actor = self._actor()
        if not isinstance(actor, FrontendBase):
            raise TypeError(f"FrontendHost expected FrontendBase, got {type(actor)!r}")
        return actor


__all__ = ["FrontendHost"]
