"""Inline-mode attach API for NEURON backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from compneurovis.adapters.base import (
    ActionBinding,
    ControlBinding,
    InlineAdapterBase,
    TraceBinding,
    emit_trace_updates,
)
from compneurovis.backends.neuron.backend import NeuronBackend
from compneurovis.core.controls import ActionSpec, ControlSpec


@dataclass
class NeuronControlBinding(ControlBinding):
    def apply(self, backend: NeuronBackend, value: Any) -> bool:
        del backend
        self.set(float(value))
        return True


class _AttachBackend(NeuronBackend):
    def __init__(
        self,
        *,
        sections: list,
        controls: list[ControlBinding],
        actions: list[ActionBinding],
        traces: list[TraceBinding],
        step_fn: Callable[[], None] | None,
        dt: float,
        v_init: float,
        title: str,
    ) -> None:
        super().__init__(dt=dt, v_init=v_init, title=title)
        self._provided_sections = sections
        self._provided_controls = controls
        self._provided_actions = actions
        self._provided_traces = traces
        self._custom_step_fn = step_fn

    def build_sections(self) -> list:
        return self._provided_sections

    def control_specs(self) -> dict[str, ControlSpec]:
        return {control._control_id: control._control_spec() for control in self._provided_controls}

    def action_specs(self) -> dict[str, ActionSpec]:
        return {action._action_id: action._action_spec() for action in self._provided_actions}

    def apply_control(self, control_id: str, value: Any) -> bool:
        for control in self._provided_controls:
            if control._control_id == control_id:
                return control.apply(self, value)
        return super().apply_control(control_id, value)

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

    def _emit_batch(self, times_array: np.ndarray, steps: list[Any]) -> None:
        super()._emit_batch(times_array, steps)
        for trace in self._provided_traces:
            trace._sample()
        emit_trace_updates(self, self._provided_traces)

    def advance(self) -> None:
        if self._custom_step_fn is None:
            super().advance()
            return
        from neuron import h

        t_target = float(h.t) + self.sim_ms_per_frame()
        steps = []
        times = []
        while True:
            self._custom_step_fn()
            times.append(float(h.t))
            steps.append(self._sample_step())
            if float(h.t) >= t_target:
                break
        self._emit_batch(np.asarray(times, dtype=np.float32), steps)

    def idle_sleep(self) -> float:
        return 1.0 / 60.0


class NeuronAttachAdapter(InlineAdapterBase):
    def __init__(
        self,
        *,
        sections: list,
        step: Callable[[], None] | None,
        dt: float,
        v_init: float,
        title: str = "CompNeuroVis",
    ) -> None:
        super().__init__(title=title)
        self._sections = sections
        self._step_fn = step
        self._dt = dt
        self._v_init = v_init

    def control(
        self,
        name: str,
        *,
        label: str,
        get: Callable[[], float],
        set: Callable[[Any], None],
        min: float = 0.0,
        max: float = 1.0,
    ) -> "NeuronAttachAdapter":
        self._add_control(
            NeuronControlBinding(
                name=name,
                label=label,
                get=get,
                set=set,
                min=min,
                max=max,
            )
        )
        return self

    def _make_backend(self) -> _AttachBackend:
        return _AttachBackend(
            sections=self._sections,
            controls=self._controls,
            actions=self._actions,
            traces=self._traces,
            step_fn=self._step_fn,
            dt=self._dt,
            v_init=self._v_init,
            title=self.title,
        )


def attach(
    *,
    sections: Sequence,
    step: Callable[[], None] | None = None,
    dt: float | None = None,
    v_init: float = -65.0,
    title: str = "CompNeuroVis",
) -> NeuronAttachAdapter:
    """Attach CompNeuroVis to an existing NEURON model."""

    from neuron import h

    resolved_dt = float(dt) if dt is not None else float(h.dt)
    return NeuronAttachAdapter(
        sections=list(sections),
        step=step,
        dt=resolved_dt,
        v_init=v_init,
        title=title,
    )


__all__ = ["NeuronAttachAdapter", "NeuronControlBinding", "attach"]
