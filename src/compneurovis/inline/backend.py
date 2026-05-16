"""Runtime backend actors for inline-mode sources."""

from __future__ import annotations

import contextvars
from typing import Any, Callable

from compneurovis.backends.base import BackendBase
from compneurovis.core.app import AppSpec
from compneurovis.core.messages import InvokeAction, Message, MessagePayload, SetControl
from compneurovis.inline.bindings import ActionBinding, ControlBinding, TraceBinding, emit_trace_updates
from compneurovis.relays.base import BackendRelayBase

_current_relay_actor: contextvars.ContextVar["ComposedBackend | None"] = contextvars.ContextVar(
    "current_relay_actor",
    default=None,
)


class SourceStepContext:
    """Per-update context for sources that produce multiple samples per tick."""

    def __init__(self, traces: list[TraceBinding]) -> None:
        self._traces = traces

    def sample(self) -> None:
        for trace in self._traces:
            trace._sample()

    def _begin_update(self) -> None:
        for trace in self._traces:
            trace._begin_frame()


class InlineBackend(BackendBase):
    """Backend actor for pure-Python inline sources."""

    _FRAME_MS = 1000.0 / 60.0

    def __init__(
        self,
        *,
        traces: list[TraceBinding],
        controls: list[ControlBinding],
        actions: list[ActionBinding],
        step: Callable[[SourceStepContext], None] | None,
    ) -> None:
        super().__init__()
        self._traces = traces
        self._controls = controls
        self._actions = actions
        self._step_fn = step
        self._step_context = SourceStepContext(traces)
        self._done = False

    def handle(self, message: Message[MessagePayload]) -> None:
        payload = message.payload
        if isinstance(payload, SetControl):
            for control in self._controls:
                if control._control_id == payload.control_id:
                    control.apply(self, payload.value)
                    break
        elif isinstance(payload, InvokeAction):
            for action in self._actions:
                if action._action_id == payload.action_id:
                    action.fn()
                    if action.resets_fields:
                        for trace in self._traces:
                            self.emit_update(trace._replace_message().payload)
                    break

    def update(self) -> None:
        if self._step_fn is not None and not self._done:
            self._step_context._begin_update()
            try:
                self._step_fn(self._step_context)
            except StopIteration:
                self._done = True
        emit_trace_updates(self, self._traces)

    def idle_sleep(self) -> float:
        return self._FRAME_MS / 1000.0


class ComposedBackend(BackendRelayBase):
    """Relay actor for controls/actions that coordinate a primary source.

    Routes messages between the frontend and one or more backend sources.
    Owns no simulation state — it forwards, dispatches, and aggregates.
    """

    def __init__(
        self,
        *,
        primary: BackendBase,
        traces: list[TraceBinding],
        controls: list[ControlBinding],
        actions: list[ActionBinding],
    ) -> None:
        super().__init__()
        self.primary = primary
        self._traces = traces
        self._controls = controls
        self._actions = actions

    def initialize(self, app_spec: AppSpec) -> None:
        self.primary.initialize(app_spec)
        self._forward_primary_messages()

    def handle(self, message: Message[MessagePayload]) -> None:
        payload = message.payload
        if isinstance(payload, SetControl):
            for control in self._controls:
                if control._control_id == payload.control_id:
                    token = _current_relay_actor.set(self)
                    try:
                        control.apply(self, payload.value)
                    finally:
                        _current_relay_actor.reset(token)
                    return
        elif isinstance(payload, InvokeAction):
            for action in self._actions:
                if action._action_id == payload.action_id:
                    token = _current_relay_actor.set(self)
                    try:
                        action.fn()
                        if action.resets_fields:
                            for trace in self._traces:
                                self.emit_update(trace._replace_message().payload)
                    finally:
                        _current_relay_actor.reset(token)
                    return

        self.primary.handle(message)
        self._forward_primary_messages()

    def update(self) -> None:
        for trace in self._traces:
            trace._begin_frame()
        if self.primary.is_live():
            self.primary.update()
        self._forward_primary_messages()
        emit_trace_updates(self, self._traces)

    def is_live(self) -> bool:
        return self.primary.is_live()

    def idle_sleep(self) -> float:
        return self.primary.idle_sleep()

    def shutdown(self) -> None:
        self.primary.shutdown()

    def _forward_primary_messages(self) -> None:
        for message in self.primary.take_outbound_messages():
            self.emit(message)


__all__ = ["ComposedBackend", "InlineBackend", "SourceStepContext", "_current_relay_actor"]
