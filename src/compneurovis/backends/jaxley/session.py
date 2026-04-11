from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from compneurovis.backends.jaxley.scene import JaxleySceneBuilder
from compneurovis.core.controls import ControlSpec
from compneurovis.core.scene import Scene
from compneurovis.core.views import LinePlotViewSpec
from compneurovis.session import BufferedSession, EntityClicked, FieldAppend, FieldReplace, HistoryCaptureMode, InvokeAction, KeyPressed, Reset, SetControl, StatePatch, Status

if TYPE_CHECKING:  # pragma: no cover - optional dependency typing only
    import jaxley as jx


class SessionInteractionContext:
    def __init__(self, session: "JaxleySession"):
        self.session = session

    def set_state(self, key: str, value: Any) -> None:
        self.session._ui_state[key] = value
        self.session.emit(StatePatch({key: value}))

    def state(self, key: str, default: Any = None) -> Any:
        return self.session._ui_state.get(key, default)

    @property
    def selected_entity_id(self) -> str | None:
        value = self.session._ui_state.get("selected_entity_id")
        return str(value) if value is not None else None

    def entity_info(self, entity_id: str | None = None) -> dict[str, Any] | None:
        current_id = entity_id or self.selected_entity_id
        if current_id is None or self.session.geometry is None:
            return None
        try:
            return self.session.geometry.entity_info(current_id)
        except KeyError:
            return None

    def show_status(self, message: str, timeout_ms: int | None = None) -> None:
        self.session.emit(Status(message, timeout_ms))

    def clear_status(self) -> None:
        self.session.emit(Status("", 0))

    def invoke_action(self, action_id: str, payload: dict[str, Any] | None = None) -> None:
        self.session._dispatch_action(action_id, payload or {})


class JaxleySession(BufferedSession, ABC):
    HISTORY_CAPTURE_ON_DEMAND = HistoryCaptureMode.ON_DEMAND
    HISTORY_CAPTURE_FULL = HistoryCaptureMode.FULL

    def __init__(
        self,
        *,
        dt: float = 0.1,
        v_init: float = -70.0,
        max_samples: int = 1000,
        display_dt: float | None = 0.1,
        history_capture_mode: HistoryCaptureMode | str = HistoryCaptureMode.ON_DEMAND,
        title: str = "CompNeuroVis",
    ):
        super().__init__()
        self.dt = dt
        self.v_init = v_init
        self.max_samples = max_samples
        self.display_dt = display_dt
        self.history_capture_mode = HistoryCaptureMode(history_capture_mode)
        self.title = title
        self.cells = None
        self.network = None
        self.geometry = None
        self._runtime_handles = None
        self._field_max_samples: dict[str, int] = {}
        self._init_fn = None
        self._step_fn = None
        self._state = None
        self._all_params = None
        self._externals: dict[str, np.ndarray] = {}
        self._external_inds: dict[str, np.ndarray] = {}
        self._rec_indices: np.ndarray | None = None
        self._rec_states: tuple[str, ...] = ()
        self._time = 0.0
        self._step_index = 0
        self._ui_state: dict[str, Any] = {}
        self._entity_index_by_id: dict[str, int] = {}
        self._last_display_values: np.ndarray | None = None
        self._last_voltage_values: np.ndarray | None = None
        self._trace_segment_ids: list[str] = []
        self._trace_history_times: list[float] = []
        self._trace_history_values_by_id: dict[str, list[float]] = {}

    @abstractmethod
    def build_cells(self) -> Iterable["jx.Cell"] | "jx.Cell":
        pass

    def build_network(self, cells: list["jx.Cell"]):
        import jax  # noqa: F401
        import jaxley as jx

        return jx.Network(cells)

    def setup_model(self, network, cells):
        del network, cells
        return None

    def cell_names(self, cells: list["jx.Cell"]) -> list[str]:
        return [str(getattr(cell, "meta_name", f"cell_{i}")) for i, cell in enumerate(cells)]

    def control_specs(self) -> dict[str, ControlSpec]:
        return {}

    def action_specs(self) -> dict[str, object]:
        return {}

    def control_order(self) -> tuple[str, ...] | None:
        return None

    def action_order(self) -> tuple[str, ...] | None:
        return None

    def trace_view_updates(self) -> dict[str, Any]:
        return {}

    def display_field_id(self) -> str:
        return JaxleySceneBuilder.DISPLAY_FIELD_ID

    def history_field_id(self) -> str:
        return JaxleySceneBuilder.HISTORY_FIELD_ID

    def display_unit(self) -> str | None:
        return "mV"

    def history_unit(self) -> str | None:
        return self.display_unit()

    def morphology_color_map(self) -> str:
        return "scalar"

    def morphology_color_limits(self) -> tuple[float, float] | None:
        return None

    def morphology_color_norm(self) -> str:
        return "auto"

    def trace_title(self) -> str:
        return "Trace"

    def trace_y_label(self) -> str:
        return "Value"

    def trace_y_unit(self) -> str:
        return self.history_unit() or ""

    def apply_control(self, control_id: str, value) -> bool:
        try:
            setattr(self, control_id, value)
            return True
        except Exception:
            return False

    def apply_action(self, action_id: str, payload: dict[str, object]) -> bool:
        del action_id, payload
        return False

    def on_action(self, action_id: str, payload: dict[str, Any], context) -> bool:
        del action_id, payload, context
        return False

    def on_key_press(self, key: str, context) -> bool:
        del key, context
        return False

    def on_entity_clicked(self, entity_id: str, context) -> bool:
        del entity_id, context
        return False

    def should_capture_trace_on_click(self, entity_id: str, context) -> bool:
        del entity_id, context
        return True

    def build_scene(self, *, geometry, display_values: np.ndarray, time_value: float) -> Scene:
        controls = self.control_specs()
        actions = self.action_specs()
        trace_segment_ids, trace_times, trace_values = self._trace_field_snapshot()
        scene = JaxleySceneBuilder.build_scene(
            geometry=geometry,
            display_values=display_values,
            trace_values=trace_values,
            trace_segment_ids=trace_segment_ids,
            trace_times=trace_times,
            display_field_id=self.display_field_id(),
            history_field_id=self.history_field_id(),
            display_unit=self.display_unit(),
            history_unit=self.history_unit(),
            morphology_color_map=self.morphology_color_map(),
            morphology_color_limits=self.morphology_color_limits(),
            morphology_color_norm=self.morphology_color_norm(),
            trace_title=self.trace_title(),
            trace_y_label=self.trace_y_label(),
            trace_y_unit=self.trace_y_unit(),
            controls=controls,
            actions=actions,
            title=self.title,
            control_ids=self.control_order(),
            action_ids=self.action_order(),
        )
        trace_updates = self.trace_view_updates()
        if trace_updates:
            scene.replace_view("trace", trace_updates)
        return scene

    def initialize(self):
        import jax  # noqa: F401
        import jaxley as jx
        from jaxley.integrate import build_init_and_step_fn

        built = self.build_cells()
        self.cells = [built] if isinstance(built, jx.Cell) else list(built)
        self.network = self.build_network(self.cells)
        self._runtime_handles = self.setup_model(self.network, self.cells)
        self.network.delete_recordings()
        self.network.record("v", verbose=False)
        self.network.to_jax()
        params = self.network.get_parameters()
        self._init_fn, self._step_fn = build_init_and_step_fn(self.network)
        self._state, self._all_params = self._init_fn(params, delta_t=self.dt)
        self._externals = {key: np.asarray(value) for key, value in self.network.externals.copy().items()}
        self._external_inds = {key: np.asarray(value) for key, value in self.network.external_inds.copy().items()}
        self._rec_indices = np.asarray(self.network.recordings.rec_index.to_numpy(), dtype=np.int32)
        self._rec_states = tuple(str(value) for value in self.network.recordings.state.to_numpy().tolist())
        self._time = 0.0
        self._step_index = 0
        self.geometry = JaxleySceneBuilder.build_morphology_geometry(
            self.network.nodes,
            xyzr=self.network.xyzr,
            cell_names=self.cell_names(self.cells),
        )
        display_values = self._read_display_values()
        self._entity_index_by_id = {entity_id: index for index, entity_id in enumerate(self.geometry.entity_ids)}
        self._initialize_trace_history(self._time, display_values)
        scene = self.build_scene(
            geometry=self.geometry,
            display_values=display_values,
            time_value=self._time,
        )
        self._field_max_samples[self.history_field_id()] = self._resolved_field_max_samples(
            scene,
            field_id=self.history_field_id(),
            append_dim="time",
        )
        self._ui_state = {}
        return scene

    def _read_display_values(self) -> np.ndarray:
        if self._rec_indices is None:
            raise RuntimeError("JaxleySession recordings are not initialized")
        values = [
            np.asarray(self._state[state_name])[int(index)]
            for state_name, index in zip(self._rec_states, self._rec_indices)
        ]
        return np.asarray(values, dtype=np.float32)

    def _read_voltage(self) -> np.ndarray:
        return self._read_display_values()

    def _initialize_trace_history(self, time_value: float, display_values: np.ndarray) -> None:
        self._last_display_values = np.asarray(display_values, dtype=np.float32)
        self._last_voltage_values = self._last_display_values
        self._trace_history_times = [float(time_value)]
        self._trace_history_values_by_id = {}
        if self.history_capture_mode == HistoryCaptureMode.FULL:
            self._trace_segment_ids = list(self.geometry.entity_ids)
            for entity_id in self._trace_segment_ids:
                index = self._entity_index_by_id[entity_id]
                self._trace_history_values_by_id[entity_id] = [float(self._last_display_values[index])]
        else:
            self._trace_segment_ids = []
            for entity_id in self._preferred_trace_entity_ids():
                self._capture_trace_entity(entity_id, include_current_sample=True)

    def _preferred_trace_entity_ids(self) -> list[str]:
        preferred: list[str] = []

        selected_trace_ids = self._ui_state.get("selected_trace_entity_ids")
        if selected_trace_ids is not None:
            for value in np.asarray(selected_trace_ids).astype(str).tolist():
                if value in self._entity_index_by_id and value not in preferred:
                    preferred.append(value)

        selected_entity_id = self._ui_state.get("selected_entity_id")
        if selected_entity_id is not None:
            value = str(selected_entity_id)
            if value in self._entity_index_by_id and value not in preferred:
                preferred.append(value)

        if not preferred and self.geometry.entity_ids:
            preferred.append(self.geometry.entity_ids[0])
        return preferred

    def _capture_trace_entity(self, entity_id: str, *, include_current_sample: bool) -> bool:
        if entity_id in self._trace_history_values_by_id:
            return False
        index = self._entity_index_by_id.get(entity_id)
        if index is None:
            return False
        history = [math.nan] * len(self._trace_history_times)
        if include_current_sample and history and self._last_display_values is not None:
            history[-1] = float(self._last_display_values[index])
        self._trace_segment_ids.append(entity_id)
        self._trace_history_values_by_id[entity_id] = history
        return True

    def _trace_field_snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        times = np.asarray(self._trace_history_times, dtype=np.float32)
        segment_ids = np.asarray(self._trace_segment_ids)
        if not self._trace_segment_ids:
            values = np.empty((0, len(self._trace_history_times)), dtype=np.float32)
        else:
            values = np.asarray(
                [self._trace_history_values_by_id[entity_id] for entity_id in self._trace_segment_ids],
                dtype=np.float32,
            )
        return segment_ids, times, values

    def _trace_field_replace(self) -> FieldReplace:
        trace_segment_ids, trace_times, trace_values = self._trace_field_snapshot()
        return FieldReplace(
            field_id=self.history_field_id(),
            values=trace_values,
            coords={
                "segment": trace_segment_ids,
                "time": trace_times,
            },
        )

    def _display_field_replace(self, display_values: np.ndarray) -> FieldReplace:
        return FieldReplace(
            field_id=self.display_field_id(),
            values=np.asarray(display_values, dtype=np.float32),
        )

    def _trim_selected_trace_history(self, max_length: int) -> None:
        if max_length < 0 or len(self._trace_history_times) <= max_length:
            return
        self._trace_history_times = self._trace_history_times[-max_length:]
        for entity_id in list(self._trace_history_values_by_id.keys()):
            self._trace_history_values_by_id[entity_id] = self._trace_history_values_by_id[entity_id][-max_length:]

    def _append_selected_trace_history(self, batch_values: np.ndarray, times: list[float]) -> None:
        if not self._trace_segment_ids:
            return
        self._trace_history_times.extend(float(time_value) for time_value in times)
        for entity_id in self._trace_segment_ids:
            index = self._entity_index_by_id[entity_id]
            self._trace_history_values_by_id[entity_id].extend(float(value) for value in batch_values[index])
        max_length = self._field_max_samples.get(self.history_field_id())
        if max_length is not None:
            self._trim_selected_trace_history(int(max_length))

    def steps_per_update(self) -> int:
        if self.display_dt is None:
            return 1
        if self.display_dt <= 0:
            raise ValueError("JaxleySession display_dt must be positive or None")
        return max(1, int(math.ceil(float(self.display_dt) / float(self.dt))))

    def _resolved_field_max_samples(self, scene: Scene, *, field_id: str, append_dim: str) -> int:
        required = int(self.max_samples)
        if self.dt <= 0:
            return required
        for view in scene.views.values():
            if not isinstance(view, LinePlotViewSpec):
                continue
            if view.field_id != field_id:
                continue
            if view.x_dim != append_dim:
                continue
            if view.rolling_window is None:
                continue
            required = max(required, int(math.ceil(float(view.rolling_window) / float(self.dt))) + 1)
        return required

    def _externals_for_step(self, step_index: int) -> dict[str, np.ndarray]:
        externals: dict[str, np.ndarray] = {}
        for key, values in self._externals.items():
            if values.ndim == 0:
                externals[key] = values
            elif values.ndim == 1:
                externals[key] = values[step_index] if step_index < values.shape[0] else np.zeros_like(values[0])
            else:
                externals[key] = values[..., step_index] if step_index < values.shape[-1] else np.zeros_like(values[..., 0])
        return externals

    def _reinitialize_runtime(self, *, preserve_state: bool) -> None:
        if self.network is None or self._init_fn is None:
            return
        # Jaxley stores authoritative mutable parameters/states in DataFrames and copies
        # them into jaxnodes/jaxedges via to_jax(). Reset and live parameter updates must
        # resync from the DataFrame-backed model before rebuilding all_params/all_states.
        self.network.to_jax()
        params = self.network.get_parameters()
        current_state = self._state if preserve_state else None
        self._state, self._all_params = self._init_fn(
            params,
            all_states=current_state,
            delta_t=self.dt,
        )

    def refresh_runtime_parameters(self, *, preserve_state: bool = True) -> None:
        self._reinitialize_runtime(preserve_state=preserve_state)

    def refresh_runtime_externals(self) -> None:
        if self.network is None:
            return
        self._externals = {key: np.asarray(value) for key, value in self.network.externals.copy().items()}
        self._external_inds = {key: np.asarray(value) for key, value in self.network.external_inds.copy().items()}

    def advance(self) -> None:
        samples: list[np.ndarray] = []
        times: list[float] = []
        for _ in range(self.steps_per_update()):
            externals = self._externals_for_step(self._step_index)
            self._state = self._step_fn(
                self._state,
                self._all_params,
                externals,
                self._external_inds,
                delta_t=self.dt,
            )
            self._step_index += 1
            self._time += float(self.dt)
            times.append(self._time)
            samples.append(self._read_display_values())

        if not samples:
            return

        batch_values = np.stack(samples, axis=1)
        self._last_display_values = np.asarray(samples[-1], dtype=np.float32)
        self._last_voltage_values = self._last_display_values

        self.emit(self._display_field_replace(self._last_display_values))

        if self.history_capture_mode == HistoryCaptureMode.FULL:
            self.emit(
                FieldAppend(
                    field_id=self.history_field_id(),
                    append_dim="time",
                    values=batch_values,
                    coord_values=np.asarray(times, dtype=np.float32),
                    max_length=self._field_max_samples.get(self.history_field_id(), self.max_samples),
                )
            )
            return

        self._append_selected_trace_history(batch_values, times)
        if self._trace_segment_ids:
            indices = [self._entity_index_by_id[entity_id] for entity_id in self._trace_segment_ids]
            self.emit(
                FieldAppend(
                    field_id=self.history_field_id(),
                    append_dim="time",
                    values=batch_values[indices, :],
                    coord_values=np.asarray(times, dtype=np.float32),
                    max_length=self._field_max_samples.get(self.history_field_id(), self.max_samples),
                )
            )

    def _interaction_context(self) -> SessionInteractionContext:
        return SessionInteractionContext(self)

    def _dispatch_action(self, action_id: str, payload: dict[str, Any]) -> bool:
        if self.on_action(action_id, payload, self._interaction_context()):
            return True
        return self.apply_action(action_id, payload)

    def handle(self, command) -> None:
        if isinstance(command, Reset):
            self._reinitialize_runtime(preserve_state=False)
            self._time = 0.0
            self._step_index = 0
            display_values = self._read_display_values()
            self._initialize_trace_history(self._time, display_values)
            self.emit(self._display_field_replace(display_values))
            self.emit(self._trace_field_replace())
        elif isinstance(command, SetControl):
            self.apply_control(command.control_id, command.value)
        elif isinstance(command, InvokeAction):
            self._dispatch_action(command.action_id, command.payload)
        elif isinstance(command, EntityClicked):
            self._ui_state["selected_entity_id"] = command.entity_id
            context = self._interaction_context()
            if self.history_capture_mode == HistoryCaptureMode.ON_DEMAND and self.should_capture_trace_on_click(command.entity_id, context):
                if self._capture_trace_entity(command.entity_id, include_current_sample=True):
                    self.emit(self._trace_field_replace())
            self.on_entity_clicked(command.entity_id, context)
        elif isinstance(command, KeyPressed):
            self.on_key_press(command.key, self._interaction_context())
