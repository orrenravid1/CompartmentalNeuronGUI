from __future__ import annotations

import queue
import time
from multiprocessing import Pipe
from multiprocessing.connection import Connection

from compneurovis.core.actor import ActorRole
from compneurovis.core.app import ActorSpec, RoutingSpec
from compneurovis.core.messages import (
    InvokeAction,
    Message,
    MessagePayload,
    RoutedCommand,
    SetControl,
    command_message,
)
from compneurovis.transports.pipe import DEFAULT_MAX_PAYLOADS_PER_POLL, DEFAULT_MAX_POLL_DURATION_S


class _RoutingMixin:
    actor_id: str
    _actor_roles: dict[str, ActorRole]
    _routing: RoutingSpec

    def _route(self, message: Message[MessagePayload]) -> tuple[tuple[str, Message[MessagePayload]], ...]:
        payload = message.payload
        if isinstance(payload, RoutedCommand):
            return ((payload.target_actor_id, command_message(payload.command)),)

        if message.intent == "update":
            targets = self._routing.default_update_targets or self._actors_with_role(ActorRole.FRONTEND)
        elif isinstance(payload, SetControl):
            targets = self._routing.control_routes.get(payload.control_id, ())
            if not targets:
                targets = self._routing.default_command_targets or self._actors_with_role(ActorRole.BACKEND)
        elif isinstance(payload, InvokeAction):
            targets = self._routing.action_routes.get(payload.action_id, ())
            if not targets:
                targets = self._routing.default_command_targets or self._actors_with_role(ActorRole.BACKEND)
        else:
            targets = self._routing.default_command_targets or self._actors_with_role(ActorRole.BACKEND)

        return tuple(
            (target_id, message)
            for target_id in targets
            if target_id != self.actor_id
        )

    def _actors_with_role(self, role: ActorRole) -> tuple[str, ...]:
        return tuple(
            actor_id
            for actor_id, actor_role in self._actor_roles.items()
            if actor_role == role
        )


class RoutedEndpoint(_RoutingMixin):
    """Pipe-backed endpoint that routes messages through actor mailboxes.

    Hosts still see the same endpoint protocol: ``send()``, ``poll()``, and
    ``close()``. Routing is transport-owned and uses actor ids, roles, and a
    generic ``RoutingSpec``.
    """

    def __init__(
        self,
        *,
        actor_id: str,
        inbound: dict[str, Connection],
        outbound: dict[str, Connection],
        actor_roles: dict[str, ActorRole],
        routing: RoutingSpec,
    ) -> None:
        self.actor_id = actor_id
        self._inbound = inbound
        self._outbound = outbound
        self._actor_roles = actor_roles
        self._routing = routing
        self.dead = False
        self.max_payloads_per_poll = DEFAULT_MAX_PAYLOADS_PER_POLL
        self.max_poll_duration_s = DEFAULT_MAX_POLL_DURATION_S

    def poll(self) -> list[Message[MessagePayload]]:
        started = time.monotonic()
        messages: list[Message[MessagePayload]] = []
        payload_count = 0
        for connection in list(self._inbound.values()):
            while not self.dead:
                if payload_count >= self.max_payloads_per_poll:
                    break
                if time.monotonic() - started >= self.max_poll_duration_s:
                    break
                try:
                    if not connection.poll():
                        break
                    payload = connection.recv()
                except (BrokenPipeError, EOFError, OSError):
                    break
                if isinstance(payload, list):
                    messages.extend(payload)
                else:
                    messages.append(payload)
                payload_count += 1
        return messages

    def send(self, message: Message[MessagePayload]) -> None:
        for target_id, routed_message in self._route(message):
            connection = self._outbound.get(target_id)
            if connection is not None:
                connection.send(routed_message)

    def close(self) -> None:
        self.dead = True
        for connection in (*self._inbound.values(), *self._outbound.values()):
            try:
                connection.close()
            except OSError:
                pass


class InProcessRoutedEndpoint(_RoutingMixin):
    """Queue-backed routed endpoint for actors that share a process."""

    def __init__(
        self,
        *,
        actor_id: str,
        inbound: queue.Queue,
        mailboxes: dict[str, queue.Queue],
        actor_roles: dict[str, ActorRole],
        routing: RoutingSpec,
    ) -> None:
        self.actor_id = actor_id
        self._inbound = inbound
        self._mailboxes = mailboxes
        self._actor_roles = actor_roles
        self._routing = routing
        self.dead = False
        self.max_payloads_per_poll = DEFAULT_MAX_PAYLOADS_PER_POLL
        self.max_poll_duration_s = DEFAULT_MAX_POLL_DURATION_S

    def poll(self) -> list[Message[MessagePayload]]:
        started = time.monotonic()
        messages: list[Message[MessagePayload]] = []
        payload_count = 0
        while not self.dead:
            if payload_count >= self.max_payloads_per_poll:
                break
            if time.monotonic() - started >= self.max_poll_duration_s:
                break
            try:
                payload = self._inbound.get_nowait()
            except queue.Empty:
                break
            if isinstance(payload, list):
                messages.extend(payload)
            else:
                messages.append(payload)
            payload_count += 1
        return messages

    def send(self, message: Message[MessagePayload]) -> None:
        for target_id, routed_message in self._route(message):
            mailbox = self._mailboxes.get(target_id)
            if mailbox is not None:
                mailbox.put(routed_message)

    def close(self) -> None:
        self.dead = True


def routed_transport(routing: RoutingSpec | None = None, *, mode: str = "pipe"):
    """TransportFactory for local routed topologies.

    ``mode="pipe"`` uses one one-way local pipe per actor pair for subprocess
    topologies. ``mode="inprocess"`` uses one queue mailbox per actor for
    thread/same-process topologies. Both keep the host endpoint protocol
    unchanged while allowing one-to-many updates, many-to-one commands, and
    explicit actor-targeted commands.
    """

    def factory(actors: list[ActorSpec]) -> dict[str, RoutedEndpoint | InProcessRoutedEndpoint]:
        actor_roles = {actor.id: actor.role for actor in actors}
        resolved_routing = routing or _default_routing(actor_roles)
        if mode == "inprocess":
            mailboxes: dict[str, queue.Queue] = {actor.id: queue.Queue() for actor in actors}
            return {
                actor.id: InProcessRoutedEndpoint(
                    actor_id=actor.id,
                    inbound=mailboxes[actor.id],
                    mailboxes=mailboxes,
                    actor_roles=actor_roles,
                    routing=resolved_routing,
                )
                for actor in actors
            }
        if mode != "pipe":
            raise ValueError(f"Unsupported routed transport mode {mode!r}. Expected 'pipe' or 'inprocess'.")

        inbound: dict[str, dict[str, Connection]] = {actor.id: {} for actor in actors}
        outbound: dict[str, dict[str, Connection]] = {actor.id: {} for actor in actors}
        for source in actors:
            for target in actors:
                if source.id == target.id:
                    continue
                target_inbound, source_outbound = Pipe(duplex=False)
                inbound[target.id][source.id] = target_inbound
                outbound[source.id][target.id] = source_outbound
        return {
            actor.id: RoutedEndpoint(
                actor_id=actor.id,
                inbound=inbound[actor.id],
                outbound=outbound[actor.id],
                actor_roles=actor_roles,
                routing=resolved_routing,
            )
            for actor in actors
        }

    return factory


def _default_routing(actor_roles: dict[str, ActorRole]) -> RoutingSpec:
    return RoutingSpec(
        default_command_targets=tuple(
            actor_id
            for actor_id, role in actor_roles.items()
            if role == ActorRole.BACKEND
        ),
        default_update_targets=tuple(
            actor_id
            for actor_id, role in actor_roles.items()
            if role == ActorRole.FRONTEND
        ),
    )


__all__ = ["RoutedEndpoint", "routed_transport"]
