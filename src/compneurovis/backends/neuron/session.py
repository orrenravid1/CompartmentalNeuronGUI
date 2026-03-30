from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any

import numpy as np
from neuron import h

from compneurovis.core.controls import ControlSpec
from compneurovis.core.document import Document
from compneurovis.core.views import LinePlotViewSpec
from compneurovis.session import BufferedSession, FieldAppend, FieldReplace, InvokeAction, Reset, SetControl
from compneurovis.backends.neuron.document import NeuronDocumentBuilder


class NeuronSession(BufferedSession, ABC):
    def __init__(
        self,
        *,
        dt: float = 0.1,
        v_init: float = -65.0,
        max_samples: int = 1000,
        display_dt: float | None = 0.5,
        title: str = "CompNeuroVis",
    ):
        super().__init__()
        self.dt = dt
        self.v_init = v_init
        self.max_samples = max_samples
        self.display_dt = display_dt
        self.title = title
        self.sections = None
        self.geometry = None
        self._segment_refs = None
        self._segment_vector = None
        self._runtime_handles = None
        self._field_max_samples: dict[str, int] = {}

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

    def build_document(self, *, geometry, voltage_values: np.ndarray, time_value: float) -> Document:
        controls = self.control_specs()
        actions = self.action_specs()
        document = NeuronDocumentBuilder.build_document(
            geometry=geometry,
            voltage_values=voltage_values,
            time_value=time_value,
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
        self._prepare_recorders()
        h.dt = self.dt
        h.finitialize(self.v_init)
        time_value, voltage_values = self._sample()
        document = self.build_document(
            geometry=self.geometry,
            voltage_values=voltage_values,
            time_value=time_value,
        )
        self._field_max_samples["voltage"] = self._resolved_field_max_samples(
            document,
            field_id="voltage",
            append_dim="time",
        )
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

        self.emit(
            FieldAppend(
                field_id="voltage",
                append_dim="time",
                values=np.stack(samples, axis=1),
                coord_values=np.asarray(times, dtype=np.float32),
                max_length=self._field_max_samples.get("voltage", self.max_samples),
            )
        )

    def handle(self, command) -> None:
        if isinstance(command, Reset):
            h.finitialize(self.v_init)
            time_value, voltage_values = self._sample()
            self.emit(
                FieldReplace(
                    field_id="voltage",
                    values=voltage_values[:, None],
                    coords={
                        "segment": np.asarray(self.geometry.entity_ids),
                        "time": np.asarray([time_value], dtype=np.float32),
                    },
                )
            )
        elif isinstance(command, SetControl):
            self.apply_control(command.control_id, command.value)
        elif isinstance(command, InvokeAction):
            self.apply_action(command.action_id, command.payload)
