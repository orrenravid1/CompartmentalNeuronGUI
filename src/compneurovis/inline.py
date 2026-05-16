"""Inline-mode public authoring API.

``source`` wraps the execution source and returns the object that owns
trace/control/action bindings. ``show`` has no arguments; it lowers the declared
source into the same RunSpec/start_app path used by the rest of CompNeuroVis.
"""

from __future__ import annotations

import contextvars
import inspect
from collections.abc import Iterator
from typing import Any, Callable

from compneurovis.adapters.base import (
    ActionBinding,
    ControlBinding,
    InlineSourceBase,
    TraceBinding,
    append_bindings_to_app_spec,
    emit_trace_updates,
)
from compneurovis.backends.base import BackendBase
from compneurovis.relays.base import BackendRelayBase
from compneurovis.core.app import (
    AppSpec,
    DataCatalog,
    InteractionCatalog,
    LayoutCatalog,
    LayoutSpec,
    PanelSpec,
    ViewCatalog,
)
from compneurovis.core.messages import InvokeAction, Message, MessagePayload, SetControl
from compneurovis.core.actor import ActorRole
from compneurovis.core.messages import CommandPayload, Reset, command_message


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


class InlineSource(InlineSourceBase):
    """Adapter for a callable or iterator-driven pure-Python source."""

    def __init__(self, source_like: Callable[..., None] | Iterator, *, title: str = "CompNeuroVis") -> None:
        super().__init__(title=title)
        self._source_like = source_like

    def _make_backend(self) -> InlineBackend:
        return InlineBackend(
            traces=self._traces,
            controls=self._controls,
            actions=self._actions,
            step=self._step_function(),
        )

    def _build_app_spec_for_backend(self, backend: BackendBase) -> AppSpec:
        del backend
        return _build_inline_app_spec(
            title=self.title,
            traces=self._traces,
            controls=self._controls,
            actions=self._actions,
        )

    def _step_function(self) -> Callable[[SourceStepContext], None] | None:
        if self._source_like is None:
            return None
        if callable(self._source_like):
            if _callable_accepts_step_context(self._source_like):
                return self._source_like

            def step_without_context(_context: SourceStepContext) -> None:
                self._source_like()

            return step_without_context
        iterator = iter(self._source_like)

        def step(_context: SourceStepContext) -> None:
            next(iterator)

        return step


class RemoteActorRef:
    """Reference to an actor hosted outside the current Python source."""

    def __init__(
        self,
        actor_id: str,
        *,
        role: ActorRole = ActorRole.BACKEND,
        send: Callable[[CommandPayload], None] | None = None,
    ) -> None:
        self.actor_id = actor_id
        self.role = role
        self._send = send

    def command(self, command: CommandPayload) -> None:
        if self._send is not None:
            self._send(command)
            return
        relay = _current_relay_actor.get()
        if relay is None:
            raise RuntimeError(
                "RemoteActorRef without a send callback can only be used while a relay "
                "control/action is being applied."
            )
        relay.emit_command_routed(self.actor_id, command)

    def set_control(self, control_id: str, value: Any) -> None:
        self.command(SetControl(control_id, value))

    def invoke_action(self, action_id: str, payload: dict[str, Any] | None = None) -> None:
        self.command(InvokeAction(action_id, payload or {}))

    def reset(self) -> None:
        self.command(Reset())


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


class ComposedSource(InlineSourceBase):
    """Source adapter for controls/actions that coordinate more than one source.

    Compiles to a ComposedBackend at runtime. At the authoring layer it is still
    a source — it supports trace/control/action accumulation just like any
    other source adapter.
    """

    def __init__(
        self,
        primary: InlineSourceBase,
        *,
        title: str | None = None,
        participants: tuple[Any, ...] = (),
    ) -> None:
        super().__init__(title=title or primary.title)
        self.primary = primary
        self.participants = participants

    def _make_backend(self) -> ComposedBackend:
        return ComposedBackend(
            primary=self.primary._make_backend(),
            traces=self._traces,
            controls=self._controls,
            actions=self._actions,
        )

    def _build_app_spec_for_backend(self, backend: BackendBase) -> AppSpec:
        if not isinstance(backend, ComposedBackend):
            raise TypeError(f"ComposedSource expected ComposedBackend, got {type(backend)!r}")
        app_spec = self.primary._build_app_spec_for_backend(backend.primary)
        return append_bindings_to_app_spec(
            app_spec,
            traces=self._traces,
            controls=self._controls,
            actions=self._actions,
        )

    def _notebook_dt(self) -> float:
        return self.primary._notebook_dt()


class RemoteSource(InlineSourceBase):
    """Source adapter for an actor hosted outside the current Python process.

    At the authoring layer this is still a source — controls and actions route
    commands to the remote actor; traces declare field subscriptions rather than
    local read lambdas. Compilation to a RunSpec connection slot is not yet
    implemented.
    """

    def __init__(self, actor_ref: RemoteActorRef, *, title: str = "CompNeuroVis") -> None:
        super().__init__(title=title)
        self._actor_ref = actor_ref

    def _make_backend(self) -> BackendBase:
        raise NotImplementedError(
            "RemoteSource does not create a local backend. "
            "Remote source compilation to RunSpec is not yet implemented."
        )

    def _build_app_spec_for_backend(self, backend: BackendBase) -> AppSpec:
        raise NotImplementedError("RemoteSource AppSpec comes from the remote actor.")


class InlineApp:
    """Internal accumulator for one module-level inline authoring session."""

    def __init__(self) -> None:
        self._sources: list[InlineSourceBase] = []

    def source(self, source_like: Any) -> InlineSourceBase:
        adapter = _coerce_source(source_like)
        self._sources.append(adapter)
        return adapter

    def compose(self, primary: Any, *participants: Any) -> ComposedSource:
        primary_adapter = _coerce_source(primary)
        participant_refs = tuple(
            participant if isinstance(participant, RemoteActorRef) else _coerce_source(participant)
            for participant in participants
        )
        wrapped = {
            id(primary_adapter),
            *(id(participant) for participant in participant_refs if isinstance(participant, InlineSourceBase)),
        }
        self._sources = [source for source in self._sources if id(source) not in wrapped]
        adapter = ComposedSource(
            primary_adapter,
            participants=participant_refs,
        )
        self._sources.append(adapter)
        return adapter

    def remote(self, actor_ref: RemoteActorRef) -> RemoteSource:
        adapter = RemoteSource(actor_ref)
        self._sources.append(adapter)
        return adapter

    def show(self):
        if not self._sources:
            raise RuntimeError("cnv.show() requires at least one cnv.source(...) call.")
        if len(self._sources) > 1:
            raise NotImplementedError(
                "Multiple cnv.source(...) calls are accepted by the authoring API, "
                "but multi-backend routing is not implemented yet."
            )
        return self._sources[0].launch()


def _build_inline_app_spec(
    *,
    title: str,
    traces: list[TraceBinding],
    controls: list[ControlBinding],
    actions: list[ActionBinding],
) -> AppSpec:
    trace_panels = [trace._panel_spec() for trace in traces]
    controls_panel = (
        PanelSpec(
            id="controls-panel",
            kind="controls",
            control_ids=tuple(control._control_id for control in controls),
            action_ids=tuple(action._action_id for action in actions),
        )
        if controls or actions
        else None
    )
    panels = tuple(trace_panels) + ((controls_panel,) if controls_panel else ())

    if controls_panel and trace_panels:
        panel_grid = tuple(
            (panel.id, controls_panel.id) if index == 0 else (panel.id,)
            for index, panel in enumerate(trace_panels)
        )
    elif controls_panel:
        panel_grid = ((controls_panel.id,),)
    else:
        panel_grid = tuple((panel.id,) for panel in trace_panels)

    app_spec = AppSpec(
        data=DataCatalog(fields={trace._field_id: trace._initial_field() for trace in traces}),
        view_catalog=ViewCatalog(views={trace._view_id: trace._view_spec() for trace in traces}),
        interactions=InteractionCatalog(
            controls={control._control_id: control._control_spec() for control in controls},
            actions={action._action_id: action._action_spec() for action in actions},
        ),
        layout_catalog=LayoutCatalog.single(
            LayoutSpec(
                title=title,
                panels=panels,
                panel_grid=panel_grid,
            )
        ),
    )
    return append_bindings_to_app_spec(app_spec, traces=[], controls=[], actions=[])


def _coerce_source(source_like: Any) -> InlineSourceBase:
    if isinstance(source_like, InlineSourceBase):
        return source_like
    if callable(source_like):
        return InlineSource(source_like)
    try:
        iterator = iter(source_like)
    except TypeError as exc:
        raise TypeError(
            "cnv.source(...) expects a callable, an iterator/generator, or a CompNeuroVis adapter."
        ) from exc
    return InlineSource(iterator)


def _callable_accepts_step_context(fn: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False

    positional_params = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    required_positional = [
        parameter
        for parameter in positional_params
        if parameter.default is inspect.Parameter.empty
    ]
    required_keyword_only = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY and parameter.default is inspect.Parameter.empty
    ]
    accepts_varargs = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )

    if accepts_varargs:
        return True
    if required_keyword_only or len(required_positional) > 1:
        raise TypeError("cnv.source(...) callables must accept either zero arguments or one step context.")
    return len(required_positional) == 1


_app = InlineApp()


def _reset_inline_session() -> None:
    global _app
    _app = InlineApp()


def source(source_like: Any) -> InlineSourceBase:
    return _app.source(source_like)


def compose(primary: Any, *participants: Any) -> ComposedSource:
    return _app.compose(primary, *participants)


def remote(actor_ref: RemoteActorRef) -> RemoteSource:
    return _app.remote(actor_ref)


def remote_actor(
    actor_id: str,
    *,
    role: ActorRole = ActorRole.BACKEND,
    send: Callable[[CommandPayload], None] | None = None,
) -> RemoteActorRef:
    return RemoteActorRef(actor_id, role=role, send=send)


def show():
    return _app.show()


__all__ = [
    "InlineBackend",
    "SourceStepContext",
    "ComposedBackend",
    "ComposedSource",
    "RemoteSource",
    "InlineSource",
    "RemoteActorRef",
    "compose",
    "remote",
    "remote_actor",
    "source",
    "show",
]
