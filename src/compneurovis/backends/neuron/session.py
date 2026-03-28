from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

import numpy as np
from neuron import h

from compneurovis.core.controls import ControlSpec
from compneurovis.session import BufferedSession, FieldUpdate, Reset, SetControl
from compneurovis.backends.neuron.document import NeuronDocumentBuilder


class NeuronSession(BufferedSession, ABC):
    def __init__(self, *, dt: float = 0.1, v_init: float = -65.0, max_samples: int = 1000, title: str = "CompNeuroVis"):
        super().__init__()
        self.dt = dt
        self.v_init = v_init
        self.max_samples = max_samples
        self.title = title
        self.sections = None
        self.geometry = None
        self._segment_refs = None
        self._segment_vector = None
        self._time_history = deque(maxlen=max_samples)
        self._voltage_history = deque(maxlen=max_samples)
        self._runtime_handles = None

    @abstractmethod
    def build_sections(self):
        pass

    def setup_model(self, sections):
        return None

    def control_specs(self) -> dict[str, ControlSpec]:
        return {}

    def apply_control(self, control_id: str, value) -> bool:
        try:
            setattr(self, control_id, value)
            return True
        except Exception:
            return False

    def initialize(self):
        self.sections = self.build_sections()
        self._runtime_handles = self.setup_model(self.sections)
        self.geometry = NeuronDocumentBuilder.build_morphology_geometry(self.sections)
        self._prepare_recorders()
        h.dt = self.dt
        h.finitialize(self.v_init)
        self._time_history.clear()
        self._voltage_history.clear()
        self._append_sample()
        return NeuronDocumentBuilder.build_document(
            geometry=self.geometry,
            voltage_values=self._voltage_history[-1],
            time_value=self._time_history[-1],
            controls=self.control_specs(),
            title=self.title,
        )

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

    def _append_sample(self) -> None:
        self._time_history.append(float(h.t))
        self._voltage_history.append(self._read_voltage())

    def advance(self) -> None:
        h.fadvance()
        self._append_sample()
        self.emit(
            FieldUpdate(
                field_id="voltage",
                values=np.stack(list(self._voltage_history), axis=1),
                coords={
                    "segment": np.asarray(self.geometry.entity_ids),
                    "time": np.asarray(list(self._time_history), dtype=np.float32),
                },
            )
        )

    def handle(self, command) -> None:
        if isinstance(command, Reset):
            h.finitialize(self.v_init)
            self._time_history.clear()
            self._voltage_history.clear()
            self._append_sample()
            self.emit(
                FieldUpdate(
                    field_id="voltage",
                    values=np.stack(list(self._voltage_history), axis=1),
                    coords={
                        "segment": np.asarray(self.geometry.entity_ids),
                        "time": np.asarray(list(self._time_history), dtype=np.float32),
                    },
                )
            )
        elif isinstance(command, SetControl):
            self.apply_control(command.control_id, command.value)

