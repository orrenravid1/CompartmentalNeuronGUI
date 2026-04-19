"""
HH section inspector - simple NEURON morphology with section-linked voltage, gating, and input-current plots.

Patterns shown:
  - programmatic pt3d morphology with one rendered entity per section for click-to-inspect traces
  - custom NeuronSession scene assembly with multiple history fields and three linked line plots
  - morphology coloring that switches between membrane voltage and the HH gates n, m, and h
  - selection-driven line plots that follow the clicked morphology section without frontend-specific code

Requires: NEURON
Run: python examples/neuron/hh_section_inspector.py
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from neuron import h

from compneurovis import (
    ControlSpec,
    DiagnosticsSpec,
    Field,
    LayoutSpec,
    LinePlotViewSpec,
    MorphologyViewSpec,
    NeuronSceneBuilder,
    NeuronSession,
    PanelSpec,
    Scene,
    StateBinding,
    build_neuron_app,
    run_app,
)
from compneurovis.session import EntityClicked, FieldAppend, FieldReplace, InvokeAction, KeyPressed, Reset, ScenePatch, SetControl, StatePatch


TITLE = "HH section inspector"
SEGMENT_DIM = "segment"
TIME_DIM = "time"
GATE_DIM = "gate"
GATE_LABELS = np.asarray(("m", "h", "n"))

DISPLAY_FIELD_ID = "hh_section_display"
VOLTAGE_HISTORY_FIELD_ID = "hh_section_voltage_history"
CURRENT_HISTORY_FIELD_ID = "hh_section_input_current_history"
GATING_HISTORY_FIELD_ID = "hh_section_gating_history"

DISPLAY_VOLTAGE = "voltage"
DISPLAY_POTASSIUM_GATE = "potassium gate (n)"
DISPLAY_SODIUM_ACTIVATION = "sodium activation (m)"
DISPLAY_SODIUM_INACTIVATION = "sodium inactivation (h)"
DISPLAY_OPTIONS = (
    DISPLAY_VOLTAGE,
    DISPLAY_POTASSIUM_GATE,
    DISPLAY_SODIUM_ACTIVATION,
    DISPLAY_SODIUM_INACTIVATION,
)
VOLTAGE_TRACE_COLOR = "#00d2be"
VOLTAGE_OPPOSING_COLOR = "#ff1060"
GATE_TRACE_COLORS = {
    "m": "#ff8c00",
    "h": "#ff50b4",
    "n": "#a000ff",
}
GATE_OPPOSING_COLORS = {
    "m": "#0080ff",
    "h": "#00aaff",
    "n": "#aaff00",
}
DISPLAY_COLOR_LIMITS = {
    DISPLAY_VOLTAGE: (-80.0, 50.0),
    DISPLAY_POTASSIUM_GATE: (-0.05, 1.05),
    DISPLAY_SODIUM_ACTIVATION: (-0.05, 1.05),
    DISPLAY_SODIUM_INACTIVATION: (-0.05, 1.05),
}
DISPLAY_COLOR_MAPS = {
    DISPLAY_VOLTAGE: f"mpl-ramp:{VOLTAGE_OPPOSING_COLOR}:{VOLTAGE_TRACE_COLOR}",
    DISPLAY_POTASSIUM_GATE: f"mpl-ramp:{GATE_OPPOSING_COLORS['n']}:{GATE_TRACE_COLORS['n']}",
    DISPLAY_SODIUM_ACTIVATION: f"mpl-ramp:{GATE_OPPOSING_COLORS['m']}:{GATE_TRACE_COLORS['m']}",
    DISPLAY_SODIUM_INACTIVATION: f"mpl-ramp:{GATE_OPPOSING_COLORS['h']}:{GATE_TRACE_COLORS['h']}",
}

ROLLING_WINDOW_MS = 40.0
LINE_PLOT_MAX_REFRESH_HZ = 8.0
MORPHOLOGY_MAX_REFRESH_HZ = 4.0


def _humanize_section_name(name: str) -> str:
    return name.replace("_", " ").title()


def _straight_section(
    *,
    name: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    diam: float,
    nseg: int,
):
    sec = h.Section(name=name)
    sec.nseg = int(nseg)
    sec.pt3dclear()
    sec.pt3dadd(float(start[0]), float(start[1]), float(start[2]), float(diam))
    sec.pt3dadd(float(end[0]), float(end[1]), float(end[2]), float(diam))
    return sec


class HHSectionInspectorSession(NeuronSession):
    def __init__(self):
        super().__init__(
            dt=0.025,
            v_init=-65.0,
            max_samples=1600,
            display_dt=0.5,
            title=TITLE,
        )
        self.morphology_quantity = DISPLAY_VOLTAGE
        self.stim_scale = 1.0
        self.sections_by_name: dict[str, object] = {}
        self.clamp_specs: list[tuple[object, str, float]] = []
        self.clamps_by_section_name: dict[str, list[object]] = {}
        self.segment_ids = np.asarray([], dtype=str)
        self.entity_section_names: tuple[str, ...] = ()
        self._history_max_samples = int(self.max_samples)
        self._voltage_refs = None
        self._voltage_vector = None
        self._m_refs = None
        self._m_vector = None
        self._h_refs = None
        self._h_vector = None
        self._n_refs = None
        self._n_vector = None
        self._history_times = np.asarray([], dtype=np.float32)
        self._voltage_history = np.empty((0, 0), dtype=np.float32)
        self._current_history = np.empty((0, 0), dtype=np.float32)
        self._gating_history = np.empty((0, len(GATE_LABELS), 0), dtype=np.float32)
        self._latest_snapshot: dict[str, np.ndarray] | None = None

    def build_sections(self):
        soma = _straight_section(
            name="soma",
            start=(-12.0, 0.0, 0.0),
            end=(12.0, 0.0, 0.0),
            diam=18.0,
            nseg=1,
        )
        apical = _straight_section(
            name="apical",
            start=(12.0, 0.0, 0.0),
            end=(112.0, 44.0, 8.0),
            diam=3.5,
            nseg=9,
        )
        basal = _straight_section(
            name="basal",
            start=(-12.0, 0.0, 0.0),
            end=(-108.0, -42.0, -8.0),
            diam=3.0,
            nseg=7,
        )
        axon_initial = _straight_section(
            name="axon_initial",
            start=(-12.0, 0.0, 0.0),
            end=(-12.0, 0.0, 56.0),
            diam=1.8,
            nseg=7,
        )
        axon_distal = _straight_section(
            name="axon_distal",
            start=(-12.0, 0.0, 56.0),
            end=(-12.0, 0.0, 186.0),
            diam=1.2,
            nseg=9,
        )

        apical.connect(soma(1.0))
        basal.connect(soma(0.0))
        axon_initial.connect(soma(0.0))
        axon_distal.connect(axon_initial(1.0))

        return [soma, apical, basal, axon_initial, axon_distal]

    def control_specs(self) -> dict[str, ControlSpec]:
        return {
            "morphology_quantity": ControlSpec(
                "morphology_quantity",
                "enum",
                "Morphology coloring",
                self.morphology_quantity,
                options=DISPLAY_OPTIONS,
                send_to_session=True,
            ),
            "stim_scale": ControlSpec(
                "stim_scale",
                "float",
                "Stimulus scale",
                self.stim_scale,
                min=0.0,
                max=1.6,
                steps=160,
                send_to_session=True,
            ),
            "display_dt": ControlSpec(
                "display_dt",
                "float",
                "Visual update interval (ms sim/update)",
                self.display_dt,
                min=self.dt,
                max=4.0,
                steps=159,
                send_to_session=True,
            ),
        }

    def control_order(self) -> tuple[str, ...] | None:
        return ("morphology_quantity", "stim_scale", "display_dt")

    def idle_sleep(self) -> float:
        # Keep the example responsive under interaction. This demo does not
        # need to run flat-out; a modest cadence cap leaves headroom for the
        # GUI thread and linked line plots.
        return 1.0 / 60.0

    def setup_model(self, sections):
        self.sections_by_name = {sec.name(): sec for sec in sections}
        for sec in sections:
            sec.insert("hh")
            sec.Ra = 100.0
            sec.cm = 1.0

        pulse_specs = {
            "soma": ((8.0, 4.0, 1.00), (42.0, 5.0, 0.85), (78.0, 4.0, 1.05)),
            "apical": ((24.0, 6.0, 0.35), (60.0, 6.0, 0.35)),
            "basal": ((50.0, 8.0, -0.18),),
        }

        self.clamp_specs = []
        self.clamps_by_section_name = {sec.name(): [] for sec in sections}
        for section_name, pulses in pulse_specs.items():
            sec = self.sections_by_name[section_name]
            for delay, dur, base_amp in pulses:
                clamp = h.IClamp(sec(0.5))
                clamp.delay = delay
                clamp.dur = dur
                clamp.amp = self.stim_scale * base_amp
                self.clamp_specs.append((clamp, section_name, float(base_amp)))
                self.clamps_by_section_name[section_name].append(clamp)
        return {"clamps": [spec[0] for spec in self.clamp_specs]}

    def apply_control(self, control_id: str, value) -> bool:
        if control_id == "display_dt":
            self.display_dt = max(self.dt, float(value))
            return True
        if control_id == "stim_scale":
            self.stim_scale = float(value)
            for clamp, _section_name, base_amp in self.clamp_specs:
                clamp.amp = self.stim_scale * base_amp
            return True
        if control_id == "morphology_quantity":
            self.morphology_quantity = str(value)
            if self._latest_snapshot is not None:
                self.emit(FieldReplace(field_id=DISPLAY_FIELD_ID, values=self._display_values(self._latest_snapshot)))
            self.emit(ScenePatch(view_updates={"morphology": {"color_map": self._display_color_map(self.morphology_quantity)}}))
            self.emit(StatePatch({"morph_color_limits": self._display_color_limits(self.morphology_quantity)}))
            return True
        return super().apply_control(control_id, value)

    def build_scene(self, *, geometry, snapshot: dict[str, np.ndarray], time_value: float) -> Scene:
        del time_value
        controls = self.control_specs()
        actions = self._resolved_action_specs()
        action_ids = self._resolved_action_order(actions) or ()

        views = {
            "morphology": MorphologyViewSpec(
                id="morphology",
                title="HH morphology",
                geometry_id=geometry.id,
                color_field_id=DISPLAY_FIELD_ID,
                entity_dim=SEGMENT_DIM,
                sample_dim=None,
                color_map=self._display_color_map(self.morphology_quantity),
                color_limits=StateBinding("morph_color_limits"),
                background_color="white",
                max_refresh_hz=MORPHOLOGY_MAX_REFRESH_HZ,
            ),
            "voltage_plot": LinePlotViewSpec(
                id="voltage_plot",
                title="Selected voltage",
                field_id=VOLTAGE_HISTORY_FIELD_ID,
                x_dim=TIME_DIM,
                selectors={SEGMENT_DIM: StateBinding("selected_entity_id")},
                x_label="Time",
                x_unit="ms",
                y_label="Voltage",
                y_unit="mV",
                pen=VOLTAGE_TRACE_COLOR,
                background_color="white",
                rolling_window=ROLLING_WINDOW_MS,
                trim_to_rolling_window=True,
                max_refresh_hz=LINE_PLOT_MAX_REFRESH_HZ,
                y_min=-85.0,
                y_max=55.0,
                x_major_tick_spacing=25.0,
                x_minor_tick_spacing=5.0,
            ),
            "gating_plot": LinePlotViewSpec(
                id="gating_plot",
                title="Selected gating variables",
                field_id=GATING_HISTORY_FIELD_ID,
                x_dim=TIME_DIM,
                series_dim=GATE_DIM,
                selectors={SEGMENT_DIM: StateBinding("selected_entity_id")},
                x_label="Time",
                x_unit="ms",
                y_label="Gate value",
                show_legend=True,
                series_colors=GATE_TRACE_COLORS,
                background_color="white",
                rolling_window=ROLLING_WINDOW_MS,
                trim_to_rolling_window=True,
                max_refresh_hz=LINE_PLOT_MAX_REFRESH_HZ,
                y_min=-0.05,
                y_max=1.05,
                x_major_tick_spacing=25.0,
                x_minor_tick_spacing=5.0,
            ),
            "input_current_plot": LinePlotViewSpec(
                id="input_current_plot",
                title="Selected input current",
                field_id=CURRENT_HISTORY_FIELD_ID,
                x_dim=TIME_DIM,
                selectors={SEGMENT_DIM: StateBinding("selected_entity_id")},
                x_label="Time",
                x_unit="ms",
                y_label="Current",
                y_unit="nA",
                pen="#2446a8",
                background_color="white",
                rolling_window=ROLLING_WINDOW_MS,
                trim_to_rolling_window=True,
                max_refresh_hz=LINE_PLOT_MAX_REFRESH_HZ,
                y_min=-0.4,
                y_max=1.8,
                x_major_tick_spacing=25.0,
                x_minor_tick_spacing=5.0,
            ),
        }

        panels = (
            PanelSpec(
                id="morphology-panel",
                kind="view_3d",
                view_ids=("morphology",),
                camera_distance=230.0,
                camera_elevation=18.0,
                camera_azimuth=28.0,
            ),
            PanelSpec(id="voltage-panel", kind="line_plot", view_ids=("voltage_plot",)),
            PanelSpec(id="gating-panel", kind="line_plot", view_ids=("gating_plot",)),
            PanelSpec(id="current-panel", kind="line_plot", view_ids=("input_current_plot",)),
            PanelSpec(
                id="controls-panel",
                kind="controls",
                control_ids=self.control_order() or tuple(controls.keys()),
                action_ids=action_ids,
            ),
        )

        return Scene(
            fields={
                DISPLAY_FIELD_ID: Field(
                    id=DISPLAY_FIELD_ID,
                    values=self._display_values(snapshot),
                    dims=(SEGMENT_DIM,),
                    coords={SEGMENT_DIM: self.segment_ids},
                ),
                VOLTAGE_HISTORY_FIELD_ID: Field(
                    id=VOLTAGE_HISTORY_FIELD_ID,
                    values=self._voltage_history,
                    dims=(SEGMENT_DIM, TIME_DIM),
                    coords={
                        SEGMENT_DIM: self.segment_ids,
                        TIME_DIM: self._history_times,
                    },
                    unit="mV",
                ),
                CURRENT_HISTORY_FIELD_ID: Field(
                    id=CURRENT_HISTORY_FIELD_ID,
                    values=self._current_history,
                    dims=(SEGMENT_DIM, TIME_DIM),
                    coords={
                        SEGMENT_DIM: self.segment_ids,
                        TIME_DIM: self._history_times,
                    },
                    unit="nA",
                ),
                GATING_HISTORY_FIELD_ID: Field(
                    id=GATING_HISTORY_FIELD_ID,
                    values=self._gating_history,
                    dims=(SEGMENT_DIM, GATE_DIM, TIME_DIM),
                    coords={
                        SEGMENT_DIM: self.segment_ids,
                        GATE_DIM: GATE_LABELS,
                        TIME_DIM: self._history_times,
                    },
                ),
            },
            geometries={geometry.id: geometry},
            views=views,
            controls=controls,
            actions=actions,
            layout=LayoutSpec(
                title=self.title,
                panels=panels,
                panel_grid=(
                    ("morphology-panel", "voltage-panel"),
                    ("gating-panel", "current-panel"),
                    ("controls-panel",),
                ),
            ),
        )

    def initialize(self):
        self.sections = self.build_sections()
        self._runtime_handles = self.setup_model(self.sections)

        geometry = NeuronSceneBuilder.build_morphology_geometry(self.sections)
        geometry = replace(
            geometry,
            labels=tuple(_humanize_section_name(name) for name in geometry.section_names),
        )
        self.geometry = geometry
        self.segment_ids = np.asarray(self.geometry.entity_ids)
        self.entity_section_names = tuple(str(name) for name in self.geometry.section_names)
        self._entity_index_by_id = {
            entity_id: index for index, entity_id in enumerate(self.geometry.entity_ids)
        }

        self._prepare_recorders()

        h.dt = self.dt
        h.finitialize(self.v_init)
        time_value, snapshot = self._sample_bundle()
        self._latest_snapshot = snapshot
        self._initialize_histories(time_value, snapshot)

        scene = self.build_scene(geometry=self.geometry, snapshot=snapshot, time_value=time_value)
        history_field_ids = (
            VOLTAGE_HISTORY_FIELD_ID,
            CURRENT_HISTORY_FIELD_ID,
            GATING_HISTORY_FIELD_ID,
        )
        for field_id in history_field_ids:
            self._field_max_samples[field_id] = self._resolved_field_max_samples(
                scene,
                field_id=field_id,
                append_dim=TIME_DIM,
            )
        self._history_max_samples = max(
            int(self._field_max_samples[field_id]) for field_id in history_field_ids
        )
        for field_id in history_field_ids:
            self._field_max_samples[field_id] = self._history_max_samples

        self._ui_state = {
            "morph_color_limits": self._display_color_limits(self.morphology_quantity),
        }
        if self.geometry.entity_ids:
            initial_entity_id = self.geometry.entity_ids[0]
            self._ui_state["selected_entity_id"] = initial_entity_id
            self._ui_state["selected_entity_label"] = self.geometry.label_for(initial_entity_id)

        self.emit(StatePatch(dict(self._ui_state)))
        return scene

    def _prepare_recorders(self) -> None:
        entity_sections = [self.sections_by_name[name] for name in self.entity_section_names]
        entity_xlocs = [float(xloc) for xloc in self.geometry.xlocs]

        self._voltage_refs = h.PtrVector(len(entity_sections))
        self._voltage_vector = h.Vector(len(entity_sections))
        self._m_refs = h.PtrVector(len(entity_sections))
        self._m_vector = h.Vector(len(entity_sections))
        self._h_refs = h.PtrVector(len(entity_sections))
        self._h_vector = h.Vector(len(entity_sections))
        self._n_refs = h.PtrVector(len(entity_sections))
        self._n_vector = h.Vector(len(entity_sections))

        for index, (section, xloc) in enumerate(zip(entity_sections, entity_xlocs)):
            segment = section(xloc)
            self._voltage_refs.pset(index, segment._ref_v)
            self._m_refs.pset(index, segment._ref_m_hh)
            self._h_refs.pset(index, segment._ref_h_hh)
            self._n_refs.pset(index, segment._ref_n_hh)

    def _read_ptr_vector(self, refs, vector) -> np.ndarray:
        refs.gather(vector)
        return np.asarray(vector.as_numpy(), dtype=np.float32).copy()

    def _read_input_current_values(self) -> np.ndarray:
        current_by_section = {
            section_name: 0.0 for section_name in self.clamps_by_section_name
        }
        for section_name, clamps in self.clamps_by_section_name.items():
            current_by_section[section_name] = sum(float(clamp.i) for clamp in clamps)
        return np.asarray(
            [current_by_section.get(section_name, 0.0) for section_name in self.entity_section_names],
            dtype=np.float32,
        )

    def _sample_bundle(self) -> tuple[float, dict[str, np.ndarray]]:
        voltage = self._read_ptr_vector(self._voltage_refs, self._voltage_vector)
        sodium_m = self._read_ptr_vector(self._m_refs, self._m_vector)
        sodium_h = self._read_ptr_vector(self._h_refs, self._h_vector)
        potassium_n = self._read_ptr_vector(self._n_refs, self._n_vector)
        input_current = self._read_input_current_values()
        gating = np.stack([sodium_m, sodium_h, potassium_n], axis=1).astype(np.float32)
        return float(h.t), {
            "voltage": voltage,
            "input_current": input_current,
            "m": sodium_m,
            "h": sodium_h,
            "n": potassium_n,
            "gating": gating,
        }

    def _display_values(self, snapshot: dict[str, np.ndarray]) -> np.ndarray:
        if self.morphology_quantity == DISPLAY_POTASSIUM_GATE:
            return snapshot["n"]
        if self.morphology_quantity == DISPLAY_SODIUM_ACTIVATION:
            return snapshot["m"]
        if self.morphology_quantity == DISPLAY_SODIUM_INACTIVATION:
            return snapshot["h"]
        return snapshot["voltage"]

    def _display_color_limits(self, quantity: str) -> tuple[float, float]:
        return DISPLAY_COLOR_LIMITS.get(quantity, DISPLAY_COLOR_LIMITS[DISPLAY_VOLTAGE])

    def _display_color_map(self, quantity: str) -> str:
        return DISPLAY_COLOR_MAPS.get(quantity, DISPLAY_COLOR_MAPS[DISPLAY_VOLTAGE])

    def _initialize_histories(self, time_value: float, snapshot: dict[str, np.ndarray]) -> None:
        self._history_times = np.asarray([time_value], dtype=np.float32)
        self._voltage_history = np.asarray(snapshot["voltage"][:, None], dtype=np.float32)
        self._current_history = np.asarray(snapshot["input_current"][:, None], dtype=np.float32)
        self._gating_history = np.asarray(snapshot["gating"][:, :, None], dtype=np.float32)

    def _append_histories(
        self,
        *,
        times: np.ndarray,
        voltage_batch: np.ndarray,
        current_batch: np.ndarray,
        gating_batch: np.ndarray,
    ) -> None:
        self._history_times = np.concatenate([self._history_times, times], axis=0)
        self._voltage_history = np.concatenate([self._voltage_history, voltage_batch], axis=1)
        self._current_history = np.concatenate([self._current_history, current_batch], axis=1)
        self._gating_history = np.concatenate([self._gating_history, gating_batch], axis=2)
        if self._history_times.shape[0] > self._history_max_samples:
            self._history_times = self._history_times[-self._history_max_samples:]
            self._voltage_history = self._voltage_history[:, -self._history_max_samples:]
            self._current_history = self._current_history[:, -self._history_max_samples:]
            self._gating_history = self._gating_history[:, :, -self._history_max_samples:]

    def _voltage_history_replace(self) -> FieldReplace:
        return FieldReplace(
            field_id=VOLTAGE_HISTORY_FIELD_ID,
            values=self._voltage_history,
            coords={
                SEGMENT_DIM: self.segment_ids,
                TIME_DIM: self._history_times,
            },
        )

    def _current_history_replace(self) -> FieldReplace:
        return FieldReplace(
            field_id=CURRENT_HISTORY_FIELD_ID,
            values=self._current_history,
            coords={
                SEGMENT_DIM: self.segment_ids,
                TIME_DIM: self._history_times,
            },
        )

    def _gating_history_replace(self) -> FieldReplace:
        return FieldReplace(
            field_id=GATING_HISTORY_FIELD_ID,
            values=self._gating_history,
            coords={
                SEGMENT_DIM: self.segment_ids,
                GATE_DIM: GATE_LABELS,
                TIME_DIM: self._history_times,
            },
        )

    def advance(self) -> None:
        voltage_frames: list[np.ndarray] = []
        current_frames: list[np.ndarray] = []
        gating_frames: list[np.ndarray] = []
        times: list[float] = []
        latest_snapshot: dict[str, np.ndarray] | None = None

        for _ in range(self.steps_per_update()):
            h.fadvance()
            time_value, snapshot = self._sample_bundle()
            times.append(time_value)
            voltage_frames.append(snapshot["voltage"])
            current_frames.append(snapshot["input_current"])
            gating_frames.append(snapshot["gating"])
            latest_snapshot = snapshot

        if latest_snapshot is None:
            return

        self._latest_snapshot = latest_snapshot
        self.emit(FieldReplace(field_id=DISPLAY_FIELD_ID, values=self._display_values(latest_snapshot)))

        times_array = np.asarray(times, dtype=np.float32)
        voltage_batch = np.stack(voltage_frames, axis=1).astype(np.float32)
        current_batch = np.stack(current_frames, axis=1).astype(np.float32)
        gating_batch = np.stack(gating_frames, axis=2).astype(np.float32)
        self._append_histories(
            times=times_array,
            voltage_batch=voltage_batch,
            current_batch=current_batch,
            gating_batch=gating_batch,
        )

        self.emit(
            FieldAppend(
                field_id=VOLTAGE_HISTORY_FIELD_ID,
                append_dim=TIME_DIM,
                values=voltage_batch,
                coord_values=times_array,
                max_length=self._field_max_samples[VOLTAGE_HISTORY_FIELD_ID],
            )
        )
        self.emit(
            FieldAppend(
                field_id=CURRENT_HISTORY_FIELD_ID,
                append_dim=TIME_DIM,
                values=current_batch,
                coord_values=times_array,
                max_length=self._field_max_samples[CURRENT_HISTORY_FIELD_ID],
            )
        )
        self.emit(
            FieldAppend(
                field_id=GATING_HISTORY_FIELD_ID,
                append_dim=TIME_DIM,
                values=gating_batch,
                coord_values=times_array,
                max_length=self._field_max_samples[GATING_HISTORY_FIELD_ID],
            )
        )

    def handle(self, command) -> None:
        if isinstance(command, Reset):
            h.finitialize(self.v_init)
            time_value, snapshot = self._sample_bundle()
            self._latest_snapshot = snapshot
            self._initialize_histories(time_value, snapshot)
            self.emit(FieldReplace(field_id=DISPLAY_FIELD_ID, values=self._display_values(snapshot)))
            self.emit(self._voltage_history_replace())
            self.emit(self._current_history_replace())
            self.emit(self._gating_history_replace())
            self.emit(StatePatch({"morph_color_limits": self._display_color_limits(self.morphology_quantity)}))
            return
        if isinstance(command, SetControl):
            self.apply_control(command.control_id, command.value)
            return
        if isinstance(command, InvokeAction):
            self._dispatch_action(command.action_id, command.payload)
            return
        if isinstance(command, EntityClicked):
            if command.entity_id in self._entity_index_by_id:
                self._ui_state["selected_entity_id"] = command.entity_id
                self._ui_state["selected_entity_label"] = self.geometry.label_for(command.entity_id)
            self.on_entity_clicked(command.entity_id, self._interaction_context())
            return
        if isinstance(command, KeyPressed):
            self.on_key_press(command.key, self._interaction_context())


app = build_neuron_app(HHSectionInspectorSession)
app.diagnostics = DiagnosticsSpec(
    perf_log_enabled=True,
    perf_log_dir=".compneurovis/perf-logs/hh-section-inspector",
)

run_app(app)
