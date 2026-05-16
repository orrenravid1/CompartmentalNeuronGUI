"""Inline-mode attach API for Jaxley backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from compneurovis.adapters.base import (
    ActionBinding,
    ControlBinding,
    InlineAdapterBase,
    TraceBinding,
    emit_trace_updates,
)
from compneurovis.backends.jaxley.backend import JaxleyBackend
from compneurovis.core.controls import ActionSpec, ControlSpec


@dataclass
class JaxleyControlBinding(ControlBinding):
    refresh_externals: bool = False
    refresh_params: bool = False

    def apply(self, backend: JaxleyBackend, value: Any) -> bool:
        self.set(float(value))
        if self.refresh_externals:
            backend.refresh_runtime_externals()
            backend._step_index = 0
        if self.refresh_params:
            backend.refresh_runtime_parameters(preserve_state=True)
        return True


class _AttachBackend(JaxleyBackend):
    def __init__(
        self,
        *,
        cells: list,
        setup_fn: Callable | None,
        controls: list[ControlBinding],
        actions: list[ActionBinding],
        traces: list[TraceBinding],
        dt: float,
        v_init: float,
        title: str,
        **kwargs,
    ) -> None:
        super().__init__(dt=dt, v_init=v_init, title=title, **kwargs)
        self._provided_cells = cells
        self._setup_fn = setup_fn
        self._provided_controls = controls
        self._provided_actions = actions
        self._provided_traces = traces

    def build_cells(self) -> Iterable:
        return self._provided_cells

    def setup_model(self, network, cells):
        if self._setup_fn is not None:
            self._setup_fn(network, cells)

    def control_specs(self) -> dict[str, ControlSpec]:
        return {control._control_id: control._control_spec() for control in self._provided_controls}

    def action_specs(self) -> dict[str, ActionSpec]:
        return {action._action_id: action._action_spec() for action in self._provided_actions}

    def apply_control(self, control_id: str, value: Any) -> bool:
        for control in self._provided_controls:
            if control._control_id == control_id:
                return control.apply(self, value)
        return False

    def on_action(self, action_id: str, payload: dict, context: Any) -> bool:
        del payload, context
        for action in self._provided_actions:
            if action._action_id == action_id:
                action.fn()
                if action.resets_fields:
                    for trace in self._provided_traces:
                        self.emit_update(trace._replace_message().payload)
                return True
        return False

    def _emit_batch(self, times_array, steps: list[Any]) -> None:
        super()._emit_batch(times_array, steps)
        for trace in self._provided_traces:
            trace._sample()
        emit_trace_updates(self, self._provided_traces)


class JaxleyAttachAdapter(InlineAdapterBase):
    def __init__(
        self,
        *,
        cells: list,
        setup: Callable | None,
        dt: float,
        v_init: float,
        backend_kwargs: dict,
        title: str = "CompNeuroVis",
    ) -> None:
        super().__init__(title=title)
        self._cells = cells
        self._setup_fn = setup
        self._dt = dt
        self._v_init = v_init
        self._backend_kwargs = backend_kwargs

    def control(
        self,
        name: str,
        *,
        label: str,
        get: Callable[[], float],
        set: Callable[[Any], None],
        min: float = 0.0,
        max: float = 1.0,
        refresh_externals: bool = False,
        refresh_params: bool = False,
    ) -> "JaxleyAttachAdapter":
        self._add_control(
            JaxleyControlBinding(
                name=name,
                label=label,
                get=get,
                set=set,
                min=min,
                max=max,
                refresh_externals=refresh_externals,
                refresh_params=refresh_params,
            )
        )
        return self

    def _make_backend(self) -> _AttachBackend:
        return _AttachBackend(
            cells=self._cells,
            setup_fn=self._setup_fn,
            controls=self._controls,
            actions=self._actions,
            traces=self._traces,
            dt=self._dt,
            v_init=self._v_init,
            title=self.title,
            **self._backend_kwargs,
        )


def attach(
    *,
    cells: Sequence,
    setup: Callable | None = None,
    dt: float = 0.025,
    v_init: float = -70.0,
    title: str = "CompNeuroVis",
    **kwargs,
) -> JaxleyAttachAdapter:
    """Attach CompNeuroVis to an existing Jaxley model."""

    return JaxleyAttachAdapter(
        cells=list(cells),
        setup=setup,
        dt=dt,
        v_init=v_init,
        backend_kwargs=kwargs,
        title=title,
    )


__all__ = ["JaxleyAttachAdapter", "JaxleyControlBinding", "attach"]
