from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from compneurovis.backends.base import BackendBase

if TYPE_CHECKING:
    from compneurovis.core.messages import CommandPayload, Message, MessagePayload, UpdatePayload


class _HasEmit(Protocol):
    def emit(self, message: Message[MessagePayload]) -> None: ...
    def emit_routed(self, target_actor_id: str, message: Message[MessagePayload]) -> None: ...


class RelayMixin:
    """Mixin that adds message-routing helpers to any actor.

    Contributes no role and no actor identity — relay-ness is detected via
    isinstance(actor, RelayMixin). Role is inherited from the concrete base
    (BackendBase for BackendRelayBase, FrontendBase for FrontendRelayBase).
    """

    def emit_routed(self: _HasEmit, target_actor_id: str, message: Message[MessagePayload]) -> None:
        """Route any message to a specific actor. Outer carrier mirrors inner intent."""
        from compneurovis.core.messages import ROUTED_MESSAGE, RoutedMessage, make_message
        self.emit(make_message(message.intent, RoutedMessage(target_actor_id=target_actor_id, message=message), message_type=ROUTED_MESSAGE))

    def emit_command_routed(self: _HasEmit, target_actor_id: str, command: CommandPayload) -> None:
        """Convenience: wrap a command payload and route it to a specific actor."""
        from compneurovis.core.messages import command_message
        self.emit_routed(target_actor_id, command_message(command))

    def emit_update_routed(self: _HasEmit, target_actor_id: str, update: UpdatePayload) -> None:
        """Convenience: wrap an update payload and route it to a specific actor."""
        from compneurovis.core.messages import update_message
        self.emit_routed(target_actor_id, update_message(update))


class BackendRelayBase(RelayMixin, BackendBase):
    """Base for backend-side relay actors.

    Sits on the backend side of the transport, routes messages to sub-backends,
    aggregates their updates toward the frontend. Role is BACKEND — relay-ness
    is detectable via isinstance(actor, RelayMixin).
    """


__all__ = ["BackendRelayBase", "RelayMixin"]
