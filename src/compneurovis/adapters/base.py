"""Shared implementation for inline-mode and source attach adapters."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Self

import numpy as np

from compneurovis.backends.base import BackendBase
from compneurovis.core.app import (
    AppSpec,
    PANEL_KIND_CONTROLS,
)
from compneurovis.core.controls import ActionSpec, ControlSpec, ScalarValueSpec
from compneurovis.core.field import Field
from compneurovis.core.messages import FieldAppend, FieldReplace, update_message
from compneurovis.core.views import LinePlotViewSpec

SeriesReaders = Callable[[], float] | dict[str, Callable[[], float]]


@dataclass
class TraceBinding:
    name: str
    read: SeriesReaders
    x: Callable[[], float]
    rolling_window: float = 500.0
    y_min: float | None = None
    y_max: float | None = None
    y_unit: str = "a.u."
    x_unit: str = "ms"
    max_samples: int = 2400
    _field_id: str = field(init=False, default="")
    _view_id: str = field(init=False, default="")
    _buf_x: list = field(init=False, default_factory=list)
    _buf_vals: list = field(init=False, default_factory=list)

    def _register(self, index: int) -> None:
        self._field_id = f"field_{index}_{self.name}"
        self._view_id = f"view_{index}_{self.name}"

    def _series(self) -> dict[str, Callable[[], float]]:
        if callable(self.read):
            return {self.name: self.read}
        return self.read

    def _sample(self) -> None:
        series = self._series()
        self._buf_x.append(self.x())
        self._buf_vals.append([fn() for fn in series.values()])

    def _drain_message(self):
        if not self._buf_x:
            return None
        xs = self._buf_x[:]
        vals = self._buf_vals[:]
        self._buf_x.clear()
        self._buf_vals.clear()
        n_series = len(self._series())
        values = np.array(vals, dtype=np.float32).reshape(len(xs), n_series).T
        return update_message(
            FieldAppend(
                field_id=self._field_id,
                append_dim="time",
                values=values,
                coord_values=np.array(xs, dtype=np.float32),
                max_length=self.max_samples,
            )
        )

    def _initial_field(self) -> Field:
        series = self._series()
        return Field(
            id=self._field_id,
            values=np.array([[fn()] for fn in series.values()], dtype=np.float32),
            dims=("series", "time"),
            coords={
                "series": np.array(list(series.keys())),
                "time": np.array([self.x()], dtype=np.float32),
            },
            unit=self.y_unit,
        )

    def _view_spec(self) -> LinePlotViewSpec:
        series = self._series()
        return LinePlotViewSpec(
            id=self._view_id,
            title=self.name,
            field_id=self._field_id,
            x_dim="time",
            series_dim="series",
            x_unit=self.x_unit,
            y_unit=self.y_unit,
            rolling_window=self.rolling_window,
            trim_to_rolling_window=True,
            y_min=self.y_min,
            y_max=self.y_max,
            show_legend=len(series) > 1,
        )

    def _panel_spec(self):
        from compneurovis.core.app import PANEL_KIND_LINE_PLOT, PanelSpec

        return PanelSpec(
            id=f"panel_{self._view_id}",
            kind=PANEL_KIND_LINE_PLOT,
            view_ids=(self._view_id,),
        )

    def _replace_message(self):
        series = self._series()
        values = np.array([[fn()] for fn in series.values()], dtype=np.float32)
        return update_message(
            FieldReplace(
                field_id=self._field_id,
                values=values,
                coords={
                    "series": np.array(list(series.keys())),
                    "time": np.array([self.x()], dtype=np.float32),
                },
            )
        )


@dataclass
class ControlBinding:
    name: str
    label: str
    get: Callable[[], float]
    set: Callable[[Any], None]
    min: float = 0.0
    max: float = 1.0
    _control_id: str = field(init=False, default="")

    def _register(self, index: int) -> None:
        self._control_id = f"ctrl_{index}_{self.name}"

    def _control_spec(self) -> ControlSpec:
        return ControlSpec(
            id=self._control_id,
            label=self.label,
            value_spec=ScalarValueSpec(default=self.get(), min=self.min, max=self.max),
            send_to_backend=True,
        )

    def apply(self, backend: BackendBase, value: Any) -> bool:
        del backend
        self.set(value)
        return True


@dataclass
class ActionBinding:
    name: str
    label: str
    fn: Callable[[], None]
    resets_fields: bool = False
    _action_id: str = field(init=False, default="")

    def _register(self, index: int) -> None:
        self._action_id = f"action_{index}_{self.name}"

    def _action_spec(self) -> ActionSpec:
        return ActionSpec(id=self._action_id, label=self.label)


def append_bindings_to_app_spec(
    app_spec: AppSpec,
    *,
    traces: list[TraceBinding],
    controls: list[ControlBinding],
    actions: list[ActionBinding],
) -> AppSpec:
    """Add generic inline bindings to an AppSpec built by any backend."""

    for trace in traces:
        app_spec.data.fields[trace._field_id] = trace._initial_field()
        app_spec.view_catalog.views[trace._view_id] = trace._view_spec()

    for control in controls:
        app_spec.interactions.controls[control._control_id] = control._control_spec()
    for action in actions:
        app_spec.interactions.actions[action._action_id] = action._action_spec()

    layout = app_spec.active_layout()
    panels = list(layout.panels)
    panel_grid = list(layout.panel_grid)
    panel_ids = {panel.id for panel in panels}

    for trace in traces:
        panel = trace._panel_spec()
        if panel.id not in panel_ids:
            panels.append(panel)
            panel_grid.append((panel.id,))
            panel_ids.add(panel.id)

    control_ids = tuple(control._control_id for control in controls)
    action_ids = tuple(action._action_id for action in actions)
    if control_ids or action_ids:
        controls_panel_index = next(
            (index for index, panel in enumerate(panels) if panel.kind == PANEL_KIND_CONTROLS),
            None,
        )
        if controls_panel_index is None:
            from compneurovis.core.app import PanelSpec

            controls_panel = PanelSpec(
                id="controls-panel",
                kind=PANEL_KIND_CONTROLS,
                control_ids=control_ids,
                action_ids=action_ids,
            )
            panels.append(controls_panel)
            panel_grid.append((controls_panel.id,))
        else:
            panel = panels[controls_panel_index]
            panels[controls_panel_index] = replace(
                panel,
                control_ids=tuple(dict.fromkeys((*panel.control_ids, *control_ids))),
                action_ids=tuple(dict.fromkeys((*panel.action_ids, *action_ids))),
            )

    layout.replace_panels(tuple(panels), tuple(panel_grid))
    return app_spec


def emit_trace_updates(backend: BackendBase, traces: list[TraceBinding]) -> None:
    for trace in traces:
        msg = trace._drain_message()
        if msg is not None:
            backend.emit_update(msg.payload)


class InlineAdapterBase:
    """Base for anything that can participate in the inline authoring mode."""

    def __init__(self, *, title: str = "CompNeuroVis") -> None:
        self.title = title
        self._traces: list[TraceBinding] = []
        self._controls: list[ControlBinding] = []
        self._actions: list[ActionBinding] = []
        self._handle = None

    def trace(self, name: str, *, read: SeriesReaders, x: Callable[[], float], **kwargs) -> Self:
        self._add_trace(TraceBinding(name=name, read=read, x=x, **kwargs))
        return self

    def control(
        self,
        name: str,
        *,
        label: str,
        get: Callable[[], float],
        set: Callable[[Any], None],
        min: float = 0.0,
        max: float = 1.0,
    ) -> Self:
        self._add_control(
            ControlBinding(
                name=name,
                label=label,
                get=get,
                set=set,
                min=min,
                max=max,
            )
        )
        return self

    def action(
        self,
        name: str,
        *,
        label: str,
        fn: Callable[[], None],
        resets_fields: bool = False,
    ) -> Self:
        self._add_action(ActionBinding(name=name, label=label, fn=fn, resets_fields=resets_fields))
        return self

    def show(self):
        return self.launch()

    def launch(self):
        from compneurovis._source_runtime import launch_source

        return launch_source(self)

    def _make_backend(self) -> BackendBase:
        raise NotImplementedError

    def _notebook_dt(self) -> float:
        backend_dt = getattr(self, "_dt", None)
        return 0.025 if backend_dt is None else float(backend_dt)

    def _build_app_spec_for_backend(self, backend: BackendBase) -> AppSpec:
        build = getattr(backend, "build_startup_app_spec", None)
        if not callable(build):
            raise TypeError(f"{type(backend).__name__} does not provide build_startup_app_spec()")
        return append_bindings_to_app_spec(
            build(),
            traces=self._traces,
            controls=self._controls,
            actions=self._actions,
        )

    def _add_trace(self, binding: TraceBinding) -> None:
        binding._register(len(self._traces))
        self._traces.append(binding)

    def _add_control(self, binding: ControlBinding) -> None:
        binding._register(len(self._controls))
        self._controls.append(binding)

    def _add_action(self, binding: ActionBinding) -> None:
        binding._register(len(self._actions))
        self._actions.append(binding)
