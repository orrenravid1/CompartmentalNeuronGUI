from __future__ import annotations

from compneurovis.backends.base import BackendBase
from compneurovis.core.actor import ActorSource
from compneurovis.core.app import AppSpec
from compneurovis.core.hosts import ActorHost
from compneurovis.core.messages import StopBackend


class BackendHost(ActorHost):
    def __init__(self, endpoint=None) -> None:
        super().__init__(endpoint=endpoint)
        self._stop_requested = False

    def start(self, actor_source: ActorSource, app_spec: AppSpec) -> BackendBase:
        actor = super().start(actor_source, app_spec)
        if not isinstance(actor, BackendBase):
            raise TypeError(f"BackendHost expected an BackendBase actor, got {type(actor)!r}")
        return actor

    def receive(self) -> None:
        actor = self._actor()
        if self.endpoint is None:
            return
        for message in self.endpoint.poll():
            if isinstance(message.payload, StopBackend):
                self._stop_requested = True
                return
            actor.handle(message)

    def step(self) -> None:
        self.receive()
        if self.should_stop():
            return
        actor = self._advanceable()
        if actor.is_live():
            actor.update()
        self.flush()

    def idle_sleep(self) -> float:
        return self._advanceable().idle_sleep()

    def should_stop(self) -> bool:
        return self._stop_requested

    def _advanceable(self) -> BackendBase:
        actor = self._actor()
        if not isinstance(actor, BackendBase):
            raise TypeError(f"BackendHost expected an BackendBase actor, got {type(actor)!r}")
        return actor


__all__ = ["BackendHost"]
