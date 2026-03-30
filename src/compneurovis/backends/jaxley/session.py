from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from compneurovis.backends.jaxley.document import JaxleyDocumentBuilder
from compneurovis.core.controls import ControlSpec
from compneurovis.core.document import Document
from compneurovis.core.views import LinePlotViewSpec
from compneurovis.session import BufferedSession, FieldAppend, FieldReplace, InvokeAction, Reset, SetControl

if TYPE_CHECKING:  # pragma: no cover - optional dependency typing only
    import jaxley as jx


class JaxleySession(BufferedSession, ABC):
    def __init__(
        self,
        *,
        dt: float = 0.1,
        v_init: float = -70.0,
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
        document = JaxleyDocumentBuilder.build_document(
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
        self.geometry = JaxleyDocumentBuilder.build_morphology_geometry(
            self.network.nodes,
            xyzr=self.network.xyzr,
            cell_names=self.cell_names(self.cells),
        )
        voltage_values = self._read_voltage()
        document = self.build_document(
            geometry=self.geometry,
            voltage_values=voltage_values,
            time_value=self._time,
        )
        self._field_max_samples["voltage"] = self._resolved_field_max_samples(
            document,
            field_id="voltage",
            append_dim="time",
        )
        return document

    def _read_voltage(self) -> np.ndarray:
        if self._rec_indices is None:
            raise RuntimeError("JaxleySession recordings are not initialized")
        values = [
            np.asarray(self._state[state_name])[int(index)]
            for state_name, index in zip(self._rec_states, self._rec_indices)
        ]
        return np.asarray(values, dtype=np.float32)

    def steps_per_update(self) -> int:
        if self.display_dt is None:
            return 1
        if self.display_dt <= 0:
            raise ValueError("JaxleySession display_dt must be positive or None")
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
            samples.append(self._read_voltage())

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
            params = self.network.get_parameters()
            self._state, self._all_params = self._init_fn(params, delta_t=self.dt)
            self._time = 0.0
            self._step_index = 0
            voltage_values = self._read_voltage()
            self.emit(
                FieldReplace(
                    field_id="voltage",
                    values=voltage_values[:, None],
                    coords={
                        "segment": np.asarray(self.geometry.entity_ids),
                        "time": np.asarray([self._time], dtype=np.float32),
                    },
                )
            )
        elif isinstance(command, SetControl):
            self.apply_control(command.control_id, command.value)
        elif isinstance(command, InvokeAction):
            self.apply_action(command.action_id, command.payload)
