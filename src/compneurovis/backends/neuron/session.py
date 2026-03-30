from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any

import numpy as np
from neuron import h

from compneurovis.core.controls import ControlSpec
from compneurovis.core.document import Document
from compneurovis.core.views import LinePlotViewSpec
from compneurovis.session import BufferedSession, EntityClicked, FieldAppend, FieldReplace, HistoryCaptureMode, InvokeAction, KeyPressed, Reset, SetControl, StatePatch, Status
from compneurovis.backends.neuron.document import NeuronDocumentBuilder


class SessionInteractionContext:
    def __init__(self, session: "NeuronSession"):
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


class NeuronSession(BufferedSession, ABC):
    HISTORY_CAPTURE_ON_DEMAND = HistoryCaptureMode.ON_DEMAND
    HISTORY_CAPTURE_FULL = HistoryCaptureMode.FULL

    def __init__(
        self,
        *,
        dt: float = 0.1,
        v_init: float = -65.0,
        max_samples: int = 1000,
        display_dt: float | None = 0.5,
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
        self.sections = None
        self.geometry = None
        self._segment_refs = None
        self._segment_vector = None
        self._runtime_handles = None
        self._field_max_samples: dict[str, int] = {}
        self._ui_state: dict[str, Any] = {}
        self._entity_index_by_id: dict[str, int] = {}
        self._last_time_value: float | None = None
        self._last_voltage_values: np.ndarray | None = None
        self._trace_segment_ids: list[str] = []
        self._trace_history_times: list[float] = []
        self._trace_history_values_by_id: dict[str, list[float]] = {}

    @abstractmethod
    def build_sections(self):
        pass

    def setup_model(self, sections):
        return None

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

    def build_document(self, *, geometry, voltage_values: np.ndarray, time_value: float) -> Document:
        controls = self.control_specs()
        actions = self.action_specs()
        trace_segment_ids, trace_times, trace_values = self._trace_field_snapshot()
        document = NeuronDocumentBuilder.build_document(
            geometry=geometry,
            display_values=voltage_values,
            trace_values=trace_values,
            trace_segment_ids=trace_segment_ids,
            trace_times=trace_times,
            controls=controls,
            actions=actions,
            title=self.title,
            control_ids=self.control_order(),
            action_ids=self.action_order(),
        )
        trace_updates = self.trace_view_updates()
        if trace_updates:
            document.replace_view("trace", trace_updates)
        return document

    def initialize(self):
        self.sections = self.build_sections()
        self._runtime_handles = self.setup_model(self.sections)
        self.geometry = NeuronDocumentBuilder.build_morphology_geometry(self.sections)
        self._entity_index_by_id = {entity_id: index for index, entity_id in enumerate(self.geometry.entity_ids)}
        self._prepare_recorders()
        h.dt = self.dt
        h.finitialize(self.v_init)
        time_value, voltage_values = self._sample()
        self._initialize_trace_history(time_value, voltage_values)
        document = self.build_document(
            geometry=self.geometry,
            voltage_values=voltage_values,
            time_value=time_value,
        )
        self._field_max_samples[NeuronDocumentBuilder.TRACE_FIELD_ID] = self._resolved_field_max_samples(
            document,
            field_id=NeuronDocumentBuilder.TRACE_FIELD_ID,
            append_dim="time",
        )
        self._ui_state = {}
        return document

    def _prepare_recorders(self):
        idx_by_name = {}
        for index, sec in enumerate(self.sections):
            idx_by_name.setdefault(sec.name(), []).append(index)

        section_lookup = {sec.name(): sec for sec in self.sections}
        entity_sections = []
        entity_xlocs = []
        for entity_id, section_name, xloc in zip(self.geometry.entity_ids, self.geometry.section_names, self.geometry.xlocs):
            del entity_id
            entity_sections.append(section_lookup[section_name])
            entity_xlocs.append(float(xloc))

        self._segment_refs = h.PtrVector(len(entity_sections))
        self._segment_vector = h.Vector(len(entity_sections))
        for i, (section, xloc) in enumerate(zip(entity_sections, entity_xlocs)):
            self._segment_refs.pset(i, section(xloc)._ref_v)

    def _read_voltage(self) -> np.ndarray:
        self._segment_refs.gather(self._segment_vector)
        return np.asarray(self._segment_vector.as_numpy(), dtype=np.float32).copy()

    def _sample(self) -> tuple[float, np.ndarray]:
        return float(h.t), self._read_voltage()

    def _initialize_trace_history(self, time_value: float, voltage_values: np.ndarray) -> None:
        self._last_time_value = float(time_value)
        self._last_voltage_values = np.asarray(voltage_values, dtype=np.float32)
        self._trace_history_times = [float(time_value)]
        self._trace_history_values_by_id = {}
        if self.history_capture_mode == HistoryCaptureMode.FULL:
            self._trace_segment_ids = list(self.geometry.entity_ids)
            for entity_id in self._trace_segment_ids:
                index = self._entity_index_by_id[entity_id]
                self._trace_history_values_by_id[entity_id] = [float(self._last_voltage_values[index])]
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
        if include_current_sample and history and self._last_voltage_values is not None:
            history[-1] = float(self._last_voltage_values[index])
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
            field_id=NeuronDocumentBuilder.TRACE_FIELD_ID,
            values=trace_values,
            coords={
                "segment": trace_segment_ids,
                "time": trace_times,
            },
        )

    def _display_field_replace(self, voltage_values: np.ndarray) -> FieldReplace:
        return FieldReplace(
            field_id=NeuronDocumentBuilder.DISPLAY_FIELD_ID,
            values=np.asarray(voltage_values, dtype=np.float32),
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
        max_length = self._field_max_samples.get(NeuronDocumentBuilder.TRACE_FIELD_ID)
        if max_length is not None:
            self._trim_selected_trace_history(int(max_length))

    def steps_per_update(self) -> int:
        if self.display_dt is None:
            return 1
        if self.display_dt <= 0:
            raise ValueError("NeuronSession display_dt must be positive or None")
        return max(1, int(math.ceil(float(self.display_dt) / float(self.dt))))

    def _resolved_field_max_samples(self, document: Document, *, field_id: str, append_dim: str) -> int:
        required = int(self.max_samples)
        if self.dt <= 0:
            return required
        for view in document.views.values():
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

    def advance(self) -> None:
        samples: list[np.ndarray] = []
        times: list[float] = []
        for _ in range(self.steps_per_update()):
            h.fadvance()
            time_value, voltage_values = self._sample()
            times.append(time_value)
            samples.append(voltage_values)

        if not samples:
            return

        batch_values = np.stack(samples, axis=1)
        self._last_time_value = float(times[-1])
        self._last_voltage_values = np.asarray(samples[-1], dtype=np.float32)

        self.emit(self._display_field_replace(self._last_voltage_values))

        if self.history_capture_mode == HistoryCaptureMode.FULL:
            self.emit(
                FieldAppend(
                    field_id=NeuronDocumentBuilder.TRACE_FIELD_ID,
                    append_dim="time",
                    values=batch_values,
                    coord_values=np.asarray(times, dtype=np.float32),
                    max_length=self._field_max_samples.get(NeuronDocumentBuilder.TRACE_FIELD_ID, self.max_samples),
                )
            )
            return

        self._append_selected_trace_history(batch_values, times)
        if self._trace_segment_ids:
            indices = [self._entity_index_by_id[entity_id] for entity_id in self._trace_segment_ids]
            self.emit(
                FieldAppend(
                    field_id=NeuronDocumentBuilder.TRACE_FIELD_ID,
                    append_dim="time",
                    values=batch_values[indices, :],
                    coord_values=np.asarray(times, dtype=np.float32),
                    max_length=self._field_max_samples.get(NeuronDocumentBuilder.TRACE_FIELD_ID, self.max_samples),
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
            h.finitialize(self.v_init)
            time_value, voltage_values = self._sample()
            self._initialize_trace_history(time_value, voltage_values)
            self.emit(
                self._display_field_replace(voltage_values)
            )
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
