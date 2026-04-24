"""
Scratch exploration of an HH -> competing ligands -> receptor/effector -> HH chain
where the downstream effector additively modulates a selected HH parameter while the
downstream neuron also receives tonic current.

What this tests:
  - upstream Hodgkin-Huxley point cell produces spikes
  - a NetCon delivers event-driven transmitter pulses into GenericLigand.NET_RECEIVE
  - a second GenericLigand acts as a continuously applied competing drug
  - both ligands compete for the same GenericReceptor with separate Kd / efficacy / decay sliders
  - receptor activation drives a SetpointRelaxEffector
  - effector output additively modulates one selected downstream HH parameter
  - downstream HH point cell also receives independent tonic current drive

Requires compiled bundled mechanisms under examples/neuron/signaling_cascade_mod/.
Compile from inside that directory (for example `nrnivmodl.bat .` on Windows or
`nrnivmodl .` on Unix/macOS).

Run: python scratch/hh_competing_ligand_parameter_modulation_exploration.py
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from neuron import h, load_mechanisms

from compneurovis import (
    AppSpec,
    AttributeRef,
    ChoiceValueSpec,
    ControlPresentationSpec,
    ControlSpec,
    Field,
    LayoutSpec,
    LinePlotViewSpec,
    PanelSpec,
    ScalarValueSpec,
    Scene,
    SeriesSpec,
    run_app,
)
from compneurovis.session import BufferedSession, FieldAppend, FieldReplace, Reset, ScenePatch, SetControl, StatePatch


h.load_file("stdrun.hoc")


TITLE = "HH ligand parameter-modulation exploration"
TIME_DIM = "time"
SERIES_DIM = "series"

POINT_LENGTH_UM = 12.6157
POINT_DIAM_UM = 12.6157
CONTINUOUS_CLAMP_DUR_MS = 1.0e9

DEFAULT_DT_MS = 0.025
DEFAULT_DISPLAY_DT_MS = 0.10
ROLLING_WINDOW_MS = 180.0
LINE_PLOT_MAX_REFRESH_HZ = 40.0

MECHANISM_DIR = Path(__file__).resolve().parents[1] / "examples" / "neuron" / "signaling_cascade_mod"

VOLTAGE_COLOR = "#00d2be"
TRANSMITTER_COLOR = "#2356b8"
DRUG_COLOR = "#d1495b"
TRANSMITTER_BOUND_COLOR = "#6fb7ff"
DRUG_BOUND_COLOR = "#ff93ae"
OCCUPANCY_COLOR = "#2f9e44"
ACTIVATION_COLOR = "#7d3cff"
EFFECTOR_S_COLOR = "#ff8c00"
EFFECTOR_S_INF_COLOR = "#222222"
BASELINE_PARAM_COLOR = "#6c757d"
PARAM_DELTA_COLOR = "#1c7c54"
EFFECTIVE_PARAM_COLOR = "#e67700"


@dataclass(frozen=True, slots=True)
class PlotSpec:
    field_id: str
    view_id: str
    panel_id: str
    title: str
    series: tuple[SeriesSpec, ...]
    y_label: str
    y_unit: str = ""
    show_legend: bool = True
    y_min: float | None = None
    y_max: float | None = None


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    key: str
    label: str
    default: float
    min_value: float
    max_value: float
    slider_steps: int
    clip_min: float
    clip_max: float
    mod_gain_label: str
    mod_gain_default: float
    mod_gain_min: float
    mod_gain_max: float
    mod_gain_steps: int


class HHPointCell:
    def __init__(
        self,
        name: str,
        *,
        clamp_amp: float,
        gnabar: float = 0.12,
        gkbar: float = 0.036,
        gl: float = 0.0003,
        el: float = -54.3,
        ena: float = 50.0,
        ek: float = -77.0,
        cm: float = 1.0,
    ) -> None:
        self.name = name
        self.clamp_amp = float(clamp_amp)
        self.gnabar = float(gnabar)
        self.gkbar = float(gkbar)
        self.gl = float(gl)
        self.el = float(el)
        self.ena = float(ena)
        self.ek = float(ek)
        self.cm = float(cm)

        self.section = None
        self.segment = None
        self.clamp = None
        self.commanded_current_na = float(clamp_amp)

    def build(self) -> None:
        self.section = h.Section(name=self.name)
        self.section.L = POINT_LENGTH_UM
        self.section.diam = POINT_DIAM_UM
        self.section.nseg = 1
        self.section.insert("hh")

        self.segment = self.section(0.5)
        self.clamp = h.IClamp(self.segment)
        self.clamp.delay = 0.0
        self.clamp.dur = CONTINUOUS_CLAMP_DUR_MS

        self.apply_runtime_parameters()
        self.set_commanded_current(self.clamp_amp)

    def apply_runtime_parameters(self) -> None:
        if self.section is None:
            return
        self.section.cm = float(self.cm)
        for seg in self.section:
            seg.gnabar_hh = float(self.gnabar)
            seg.gkbar_hh = float(self.gkbar)
            seg.gl_hh = float(self.gl)
            seg.el_hh = float(self.el)
            seg.ena = float(self.ena)
            seg.ek = float(self.ek)

    def set_commanded_current(self, amplitude_na: float) -> None:
        self.commanded_current_na = float(amplitude_na)
        if self.clamp is None:
            return
        self.clamp.delay = 0.0
        self.clamp.dur = CONTINUOUS_CLAMP_DUR_MS
        self.clamp.amp = self.commanded_current_na

    @property
    def voltage(self) -> float:
        if self.segment is None:
            return -65.0
        return float(self.segment.v)

    @property
    def input_current(self) -> float:
        return float(self.commanded_current_na)


DOWNSTREAM_PARAMETER_SPECS = (
    ParameterSpec("gnabar", "Downstream gNa (S/cm^2)", 0.12, 0.01, 0.30, 290, 0.0, 0.35, "Additive gNa modulation (S/cm^2 / effector)", 0.08, -0.12, 0.12, 240),
    ParameterSpec("gkbar", "Downstream gK (S/cm^2)", 0.036, 0.005, 0.10, 190, 0.0, 0.12, "Additive gK modulation (S/cm^2 / effector)", 0.03, -0.05, 0.05, 200),
    ParameterSpec("gl", "Downstream gL (S/cm^2)", 0.0003, 0.00005, 0.005, 220, 0.0, 0.01, "Additive gL modulation (S/cm^2 / effector)", 0.001, -0.003, 0.003, 240),
    ParameterSpec("el", "Downstream EL (mV)", -54.3, -80.0, -30.0, 250, -90.0, -20.0, "Additive EL modulation (mV / effector)", 12.0, -25.0, 25.0, 250),
    ParameterSpec("ena", "Downstream ENa (mV)", 50.0, 35.0, 80.0, 225, 20.0, 90.0, "Additive ENa modulation (mV / effector)", 8.0, -20.0, 20.0, 240),
    ParameterSpec("ek", "Downstream EK (mV)", -77.0, -110.0, -50.0, 240, -120.0, -40.0, "Additive EK modulation (mV / effector)", 8.0, -20.0, 20.0, 240),
    ParameterSpec("cm", "Downstream Cm (uF/cm^2)", 1.0, 0.2, 3.0, 280, 0.1, 4.0, "Additive Cm modulation (uF/cm^2 / effector)", 0.8, -2.0, 2.0, 240),
)

DOWNSTREAM_PARAMETER_MAP = {spec.key: spec for spec in DOWNSTREAM_PARAMETER_SPECS}
DOWNSTREAM_PARAMETER_KEYS = tuple(spec.key for spec in DOWNSTREAM_PARAMETER_SPECS)


UPSTREAM_VOLTAGE_PLOT = PlotSpec(
    field_id="upstream_voltage",
    view_id="upstream_voltage_plot",
    panel_id="upstream_voltage_panel",
    title="Upstream HH voltage",
    series=(SeriesSpec("upstream_voltage", "Voltage", source=AttributeRef("upstream", "voltage"), color=VOLTAGE_COLOR),),
    y_label="Voltage",
    y_unit="mV",
    show_legend=False,
    y_min=-90.0,
    y_max=60.0,
)

COMPETITION_PLOT = PlotSpec(
    field_id="competition",
    view_id="competition_plot",
    panel_id="competition_panel",
    title="Ligands and receptor competition",
    series=(
        SeriesSpec("transmitter_C", "Transmitter ligand", source=AttributeRef("pulse_ligand", "C"), color=TRANSMITTER_COLOR),
        SeriesSpec("drug_C", "Drug ligand", source=AttributeRef("drug_ligand", "C"), color=DRUG_COLOR),
        SeriesSpec("bound1", "Bound transmitter", source=AttributeRef("receptor", "bound1"), color=TRANSMITTER_BOUND_COLOR),
        SeriesSpec("bound2", "Bound drug", source=AttributeRef("receptor", "bound2"), color=DRUG_BOUND_COLOR),
        SeriesSpec("occupancy", "Occupancy", source=AttributeRef("receptor", "occupancy"), color=OCCUPANCY_COLOR),
        SeriesSpec("activation", "Activation", source=AttributeRef("receptor", "activation"), color=ACTIVATION_COLOR),
    ),
    y_label="Concentration / occupancy",
)

EFFECTOR_PLOT = PlotSpec(
    field_id="effector",
    view_id="effector_plot",
    panel_id="effector_panel",
    title="Effector state",
    series=(
        SeriesSpec("effector_drive", "Drive", source=AttributeRef("receptor", "activation"), color=ACTIVATION_COLOR),
        SeriesSpec("effector_s", "s", source=AttributeRef("effector", "s"), color=EFFECTOR_S_COLOR),
        SeriesSpec("effector_s_inf", "s_inf", source=AttributeRef("effector", "s_inf"), color=EFFECTOR_S_INF_COLOR),
    ),
    y_label="Signal",
    y_min=-0.05,
    y_max=1.2,
)

MODULATION_PLOT = PlotSpec(
    field_id="modulation",
    view_id="modulation_plot",
    panel_id="modulation_panel",
    title="Downstream parameter modulation",
    series=(
        SeriesSpec("parameter_base", "Baseline value", source=AttributeRef("runtime", "downstream_target_base"), color=BASELINE_PARAM_COLOR),
        SeriesSpec("parameter_delta", "Added delta", source=AttributeRef("runtime", "downstream_target_delta"), color=PARAM_DELTA_COLOR),
        SeriesSpec("parameter_effective", "Effective value", source=AttributeRef("runtime", "downstream_target_value"), color=EFFECTIVE_PARAM_COLOR),
    ),
    y_label="Selected parameter value",
)

DOWNSTREAM_VOLTAGE_PLOT = PlotSpec(
    field_id="downstream_voltage",
    view_id="downstream_voltage_plot",
    panel_id="downstream_voltage_panel",
    title="Downstream HH voltage",
    series=(SeriesSpec("downstream_voltage", "Voltage", source=AttributeRef("downstream", "voltage"), color=VOLTAGE_COLOR),),
    y_label="Voltage",
    y_unit="mV",
    show_legend=False,
    y_min=-90.0,
    y_max=60.0,
)

PLOTS = (
    UPSTREAM_VOLTAGE_PLOT,
    COMPETITION_PLOT,
    EFFECTOR_PLOT,
    MODULATION_PLOT,
    DOWNSTREAM_VOLTAGE_PLOT,
)


def float_control(
    control_id: str,
    label: str,
    default: float,
    min_value: float,
    max_value: float,
    steps: int,
    target: AttributeRef | None = None,
    *,
    scale: str = "linear",
) -> ControlSpec:
    return ControlSpec(
        id=control_id,
        label=label,
        value_spec=ScalarValueSpec(default=default, min=min_value, max=max_value, value_type="float"),
        presentation=ControlPresentationSpec(kind="slider", steps=steps, scale=scale),
        send_to_session=True,
        target=target,
    )


def int_control(
    control_id: str,
    label: str,
    default: int,
    min_value: int,
    max_value: int,
    target: AttributeRef | None = None,
) -> ControlSpec:
    return ControlSpec(
        id=control_id,
        label=label,
        value_spec=ScalarValueSpec(default=default, min=min_value, max=max_value, value_type="int"),
        send_to_session=True,
        target=target,
    )


def choice_control(
    control_id: str,
    label: str,
    default: str,
    options: tuple[str, ...],
    target: AttributeRef | None = None,
) -> ControlSpec:
    return ControlSpec(
        id=control_id,
        label=label,
        value_spec=ChoiceValueSpec(default=default, options=options),
        send_to_session=True,
        target=target,
    )


def downstream_parameter_controls() -> tuple[ControlSpec, ...]:
    controls: list[ControlSpec] = []
    for spec in DOWNSTREAM_PARAMETER_SPECS:
        controls.append(
            float_control(
                f"downstream_{spec.key}",
                spec.label,
                spec.default,
                spec.min_value,
                spec.max_value,
                spec.slider_steps,
                AttributeRef("downstream_base", spec.key),
            )
        )
    return tuple(controls)


def modulation_gain_control(target_key: str) -> ControlSpec:
    spec = DOWNSTREAM_PARAMETER_MAP[target_key]
    return float_control(
        "downstream_mod_gain",
        spec.mod_gain_label,
        spec.mod_gain_default,
        spec.mod_gain_min,
        spec.mod_gain_max,
        spec.mod_gain_steps,
        AttributeRef("runtime", "downstream_mod_gain"),
    )


STATIC_CONTROLS = (
    float_control(
        "celsius",
        "Temperature (degC)",
        6.3,
        3.0,
        37.0,
        340,
        AttributeRef("runtime", "celsius"),
    ),
    float_control(
        "upstream_clamp_amp",
        "Upstream IClamp amplitude (nA)",
        0.18,
        -0.05,
        0.40,
        225,
        AttributeRef("upstream", "clamp_amp"),
    ),
    float_control(
        "pulse_event_weight",
        "Transmitter event weight (uM)",
        2.5,
        0.0,
        5.0,
        250,
        AttributeRef("runtime", "pulse_event_weight"),
    ),
    float_control(
        "pulse_threshold",
        "Spike detection threshold (mV)",
        0.0,
        -20.0,
        20.0,
        160,
        AttributeRef("runtime", "pulse_threshold"),
    ),
    float_control(
        "pulse_decay_rate",
        "Transmitter decay (/ms)",
        0.08,
        0.001,
        0.5,
        200,
        AttributeRef("pulse_ligand", "decay_rate"),
        scale="log",
    ),
    float_control(
        "drug_external_input",
        "Drug external_input (uM/ms)",
        0.0,
        0.0,
        0.08,
        200,
        AttributeRef("drug_ligand", "external_input"),
    ),
    float_control(
        "drug_decay_rate",
        "Drug decay (/ms)",
        0.03,
        0.001,
        0.5,
        200,
        AttributeRef("drug_ligand", "decay_rate"),
        scale="log",
    ),
    float_control(
        "capacity",
        "Receptor capacity",
        1.0,
        0.1,
        2.0,
        190,
        AttributeRef("receptor", "capacity"),
    ),
    float_control(
        "baseline_activity",
        "Receptor baseline_activity",
        0.0,
        0.0,
        1.0,
        200,
        AttributeRef("receptor", "baseline_activity"),
    ),
    float_control(
        "kd1",
        "Transmitter Kd (uM)",
        0.75,
        0.01,
        10.0,
        200,
        AttributeRef("receptor", "kd1"),
        scale="log",
    ),
    float_control(
        "efficacy1",
        "Transmitter efficacy",
        1.0,
        -1.0,
        2.0,
        240,
        AttributeRef("receptor", "efficacy1"),
    ),
    float_control(
        "decay1",
        "Transmitter bound decay (/ms)",
        0.12,
        0.001,
        1.0,
        200,
        AttributeRef("receptor", "decay1"),
        scale="log",
    ),
    float_control(
        "kd2",
        "Drug Kd (uM)",
        0.60,
        0.01,
        10.0,
        200,
        AttributeRef("receptor", "kd2"),
        scale="log",
    ),
    float_control(
        "efficacy2",
        "Drug efficacy",
        0.0,
        -1.0,
        2.0,
        240,
        AttributeRef("receptor", "efficacy2"),
    ),
    float_control(
        "decay2",
        "Drug bound decay (/ms)",
        0.06,
        0.001,
        1.0,
        200,
        AttributeRef("receptor", "decay2"),
        scale="log",
    ),
    float_control(
        "s_min",
        "Effector s_min",
        0.0,
        0.0,
        1.0,
        100,
        AttributeRef("effector", "s_min"),
    ),
    float_control(
        "s_max",
        "Effector s_max",
        1.0,
        0.0,
        2.0,
        120,
        AttributeRef("effector", "s_max"),
    ),
    float_control(
        "K",
        "Effector K",
        0.30,
        0.01,
        10.0,
        200,
        AttributeRef("effector", "K"),
        scale="log",
    ),
    int_control(
        "n",
        "Effector Hill coefficient n",
        3,
        1,
        6,
        AttributeRef("effector", "n"),
    ),
    float_control(
        "tau_on",
        "Effector tau_on (ms)",
        12.0,
        1.0,
        200.0,
        200,
        AttributeRef("effector", "tau_on"),
    ),
    float_control(
        "tau_off",
        "Effector tau_off (ms)",
        45.0,
        1.0,
        300.0,
        200,
        AttributeRef("effector", "tau_off"),
    ),
    float_control(
        "downstream_tonic_current",
        "Downstream tonic current (nA)",
        0.06,
        -0.05,
        0.40,
        225,
        AttributeRef("runtime", "downstream_tonic_current"),
    ),
    choice_control(
        "downstream_mod_target",
        "Downstream modulation target",
        "el",
        DOWNSTREAM_PARAMETER_KEYS,
        AttributeRef("runtime", "downstream_mod_target"),
    ),
) + downstream_parameter_controls()

CONTROL_BY_ID = {control.id: control for control in STATIC_CONTROLS}


def control_specs(
    *,
    display_dt: float = DEFAULT_DISPLAY_DT_MS,
    dt: float = DEFAULT_DT_MS,
    downstream_mod_target: str = "el",
) -> dict[str, ControlSpec]:
    controls: dict[str, ControlSpec] = {
        "display_dt": ControlSpec(
            id="display_dt",
            label="Visual update interval (ms sim/update)",
            value_spec=ScalarValueSpec(default=max(float(dt), float(display_dt)), min=float(dt), max=5.0, value_type="float"),
            presentation=ControlPresentationSpec(kind="slider", steps=199, scale="log"),
            send_to_session=True,
        )
    }
    controls.update({control.id: control for control in STATIC_CONTROLS})
    controls["downstream_mod_gain"] = modulation_gain_control(downstream_mod_target)
    return controls


def ensure_bundled_mechanisms_loaded() -> None:
    if load_mechanisms(str(MECHANISM_DIR), warn_if_already_loaded=False):
        return
    raise RuntimeError(
        "Bundled NEURON mechanisms are not compiled. "
        f"From inside '{MECHANISM_DIR}', compile .mod files so NEURON can load "
        "nrnmech.dll on Windows or platform-specific libnrnmech.* on Unix/macOS."
    )


def zero_series_history(series: tuple[SeriesSpec, ...]) -> dict[str, list[float]]:
    return {entry.key: [0.0] for entry in series}


def build_field(
    *,
    plot: PlotSpec,
    time_history: list[float],
    series_history: dict[str, list[float]],
) -> Field:
    return Field(
        id=plot.field_id,
        values=np.asarray([series_history[entry.key] for entry in plot.series], dtype=np.float32),
        dims=(SERIES_DIM, TIME_DIM),
        coords={
            SERIES_DIM: np.asarray([entry.label for entry in plot.series]),
            TIME_DIM: np.asarray(time_history, dtype=np.float32),
        },
        unit=plot.y_unit or "a.u.",
    )


def build_view(plot: PlotSpec) -> LinePlotViewSpec:
    return LinePlotViewSpec(
        id=plot.view_id,
        title=plot.title,
        field_id=plot.field_id,
        x_dim=TIME_DIM,
        series_dim=SERIES_DIM,
        x_label="Time",
        x_unit="ms",
        y_label=plot.y_label,
        y_unit=plot.y_unit,
        show_legend=plot.show_legend,
        series_colors={entry.label: entry.color for entry in plot.series},
        background_color="white",
        rolling_window=ROLLING_WINDOW_MS,
        trim_to_rolling_window=True,
        max_refresh_hz=LINE_PLOT_MAX_REFRESH_HZ,
        y_min=plot.y_min,
        y_max=plot.y_max,
        x_major_tick_spacing=40.0,
        x_minor_tick_spacing=10.0,
    )


def build_scene(
    *,
    time_history: list[float],
    histories: dict[str, dict[str, list[float]]],
    controls: dict[str, ControlSpec],
) -> Scene:
    fields = {
        plot.field_id: build_field(plot=plot, time_history=time_history, series_history=histories[plot.field_id])
        for plot in PLOTS
    }
    views = {plot.view_id: build_view(plot) for plot in PLOTS}
    return Scene(
        fields=fields,
        geometries={},
        views=views,
        controls=controls,
        layout=LayoutSpec(
            title=TITLE,
            panels=(
                PanelSpec(id=UPSTREAM_VOLTAGE_PLOT.panel_id, kind="line_plot", view_ids=(UPSTREAM_VOLTAGE_PLOT.view_id,)),
                PanelSpec(id=COMPETITION_PLOT.panel_id, kind="line_plot", view_ids=(COMPETITION_PLOT.view_id,)),
                PanelSpec(id=EFFECTOR_PLOT.panel_id, kind="line_plot", view_ids=(EFFECTOR_PLOT.view_id,)),
                PanelSpec(id=MODULATION_PLOT.panel_id, kind="line_plot", view_ids=(MODULATION_PLOT.view_id,)),
                PanelSpec(id=DOWNSTREAM_VOLTAGE_PLOT.panel_id, kind="line_plot", view_ids=(DOWNSTREAM_VOLTAGE_PLOT.view_id,)),
                PanelSpec(id="controls_panel", kind="controls", control_ids=tuple(controls.keys())),
            ),
            panel_grid=(
                (COMPETITION_PLOT.panel_id, EFFECTOR_PLOT.panel_id),
                (UPSTREAM_VOLTAGE_PLOT.panel_id, MODULATION_PLOT.panel_id),
                (DOWNSTREAM_VOLTAGE_PLOT.panel_id, "controls_panel"),
            ),
        ),
    )


class HHLigandParameterModulationSession(BufferedSession):
    @classmethod
    def startup_scene(cls) -> Scene | None:
        return build_scene(
            time_history=[0.0],
            histories={plot.field_id: zero_series_history(plot.series) for plot in PLOTS},
            controls=control_specs(downstream_mod_target="el"),
        )

    def __init__(
        self,
        *,
        dt_ms: float = DEFAULT_DT_MS,
        display_dt_ms: float = DEFAULT_DISPLAY_DT_MS,
        v_init: float = -65.0,
        max_samples: int = 8000,
    ) -> None:
        super().__init__()
        self.dt_ms = float(dt_ms)
        self.display_dt_ms = max(self.dt_ms, float(display_dt_ms))
        self.v_init = float(v_init)
        self.max_samples = int(max_samples)

        self.runtime = SimpleNamespace(
            celsius=6.3,
            pulse_event_weight=2.5,
            pulse_threshold=0.0,
            downstream_tonic_current=0.06,
            downstream_mod_target="el",
            downstream_mod_gain=modulation_gain_control("el").default_value(),
            downstream_target_base=-54.3,
            downstream_target_delta=0.0,
            downstream_target_value=-54.3,
        )

        self.upstream = HHPointCell("upstream_hh", clamp_amp=0.18)
        self.downstream = HHPointCell("downstream_hh", clamp_amp=0.06)
        self.downstream_base = SimpleNamespace(**{spec.key: spec.default for spec in DOWNSTREAM_PARAMETER_SPECS})

        self.cascade_section = None
        self.pulse_ligand = None
        self.drug_ligand = None
        self.receptor = None
        self.effector = None
        self.upstream_to_pulse_ligand = None

        self.parameters = {
            "display_dt": self.display_dt_ms,
            **{control.id: control.default_value() for control in STATIC_CONTROLS},
            "downstream_mod_gain": modulation_gain_control("el").default_value(),
        }

        self._time_history: deque[float] = deque(maxlen=self.max_samples)
        self._histories: dict[str, dict[str, deque[float]]] = {}

    def initialize(self) -> Scene:
        ensure_bundled_mechanisms_loaded()
        self._build_model()
        self._reset_simulation()
        return build_scene(
            time_history=list(self._time_history),
            histories=self._history_lists(),
            controls=self.control_specs(),
        )

    def control_specs(self) -> dict[str, ControlSpec]:
        return control_specs(
            display_dt=self.display_dt_ms,
            dt=self.dt_ms,
            downstream_mod_target=str(self.runtime.downstream_mod_target),
        )

    def idle_sleep(self) -> float:
        return 1.0 / 60.0

    def advance(self) -> None:
        step_count = self._steps_per_update()
        time_values = np.empty(step_count, dtype=np.float32)
        batch_values = {
            plot.field_id: np.empty((len(plot.series), step_count), dtype=np.float32)
            for plot in PLOTS
        }

        for step_index in range(step_count):
            self._apply_downstream_modulation()
            h.fadvance()
            sample_time, sample_values = self._capture_sample()
            time_values[step_index] = sample_time
            for plot in PLOTS:
                batch_values[plot.field_id][:, step_index] = sample_values[plot.field_id]

        for update in self._field_appends(time_values=time_values, batch_values=batch_values):
            self.emit(update)

    def handle(self, command) -> None:
        if isinstance(command, Reset):
            self._reset_simulation()
            for update in self._field_replaces():
                self.emit(update)
            return
        if isinstance(command, SetControl):
            self.apply_control(command.control_id, command.value)

    def shutdown(self) -> None:
        self.cascade_section = None
        self.pulse_ligand = None
        self.drug_ligand = None
        self.receptor = None
        self.effector = None
        self.upstream_to_pulse_ligand = None

    def _build_model(self) -> None:
        self.upstream.build()
        self.downstream.build()

        self.cascade_section = h.Section(name="cascade")
        self.cascade_section.L = 10.0
        self.cascade_section.diam = 10.0
        self.cascade_section.nseg = 1
        self.cascade_section.insert("pas")

        self.pulse_ligand = h.GenericLigand(self.cascade_section(0.5))
        self.drug_ligand = h.GenericLigand(self.cascade_section(0.5))
        self.receptor = h.GenericReceptor(self.cascade_section(0.5))
        self.effector = h.SetpointRelaxEffector(self.cascade_section(0.5))

        self.pulse_ligand.C_init = 0.0
        self.pulse_ligand.external_input = 0.0
        self.drug_ligand.C_init = 0.0
        self.receptor.n_ligands = 2

        h.setpointer(self.pulse_ligand._ref_C, "C_lig1", self.receptor)
        h.setpointer(self.drug_ligand._ref_C, "C_lig2", self.receptor)
        h.setpointer(self.receptor._ref_activation, "drive", self.effector)

        self.upstream_to_pulse_ligand = h.NetCon(self.upstream.segment._ref_v, self.pulse_ligand, sec=self.upstream.section)
        self.upstream_to_pulse_ligand.delay = 0.0

        self._reset_history()
        self._apply_all_parameters_to_model()

    def _coerce_control_value(self, control: ControlSpec, value):
        if isinstance(control.value_spec, ChoiceValueSpec):
            return str(value)
        if isinstance(control.value_spec, ScalarValueSpec) and control.value_spec.value_type == "int":
            return int(round(float(value)))
        return float(value)

    def _apply_all_parameters_to_model(self) -> None:
        for control in STATIC_CONTROLS:
            coerced = self._coerce_control_value(control, self.parameters[control.id])
            if control.target is not None and getattr(self, control.target.owner, None) is not None:
                control.target.write(self, coerced)
        dynamic_gain = modulation_gain_control(str(self.parameters["downstream_mod_target"]))
        dynamic_gain_value = self._coerce_control_value(dynamic_gain, self.parameters["downstream_mod_gain"])
        dynamic_gain.target.write(self, dynamic_gain_value)

        self.display_dt_ms = max(self.dt_ms, float(self.parameters["display_dt"]))
        self._apply_runtime_parameters()

    def _apply_runtime_parameters(self) -> None:
        h.celsius = float(self.runtime.celsius)
        self.upstream.apply_runtime_parameters()
        self.upstream.set_commanded_current(self.upstream.clamp_amp)
        self._apply_runtime_links()
        self._apply_downstream_modulation()

    def _apply_runtime_links(self) -> None:
        if self.upstream_to_pulse_ligand is None:
            return
        self.upstream_to_pulse_ligand.threshold = float(self.runtime.pulse_threshold)
        self.upstream_to_pulse_ligand.weight[0] = float(self.runtime.pulse_event_weight)

    def _apply_downstream_modulation(self) -> None:
        if self.downstream is None:
            return

        for key in DOWNSTREAM_PARAMETER_KEYS:
            setattr(self.downstream, key, float(getattr(self.downstream_base, key)))

        target_key = str(self.runtime.downstream_mod_target)
        target_spec = DOWNSTREAM_PARAMETER_MAP[target_key]
        base_value = float(getattr(self.downstream_base, target_key))
        effector_value = 0.0 if self.effector is None else float(self.effector.effect)
        effective_value = base_value + float(self.runtime.downstream_mod_gain) * effector_value
        effective_value = min(target_spec.clip_max, max(target_spec.clip_min, effective_value))

        setattr(self.downstream, target_key, effective_value)
        self.downstream.apply_runtime_parameters()
        self.downstream.set_commanded_current(float(self.runtime.downstream_tonic_current))

        self.runtime.downstream_target_base = base_value
        self.runtime.downstream_target_delta = effective_value - base_value
        self.runtime.downstream_target_value = effective_value

    def _reset_history(self) -> None:
        self._time_history = deque(maxlen=self.max_samples)
        self._histories = {
            plot.field_id: {entry.key: deque(maxlen=self.max_samples) for entry in plot.series}
            for plot in PLOTS
        }

    def _reset_simulation(self) -> None:
        self._apply_runtime_parameters()
        h.dt = float(self.dt_ms)
        h.finitialize(float(self.v_init))
        self._reset_history()
        self._apply_downstream_modulation()
        self._capture_sample()

    def _steps_per_update(self) -> int:
        return max(1, int(math.ceil(self.display_dt_ms / self.dt_ms)))

    def _read_plot_values(self, plot: PlotSpec) -> np.ndarray:
        return np.asarray([float(entry.source.read(self)) for entry in plot.series], dtype=np.float32)

    def _capture_sample(self) -> tuple[float, dict[str, np.ndarray]]:
        time_value = float(h.t)
        self._time_history.append(time_value)

        sample_values: dict[str, np.ndarray] = {}
        for plot in PLOTS:
            values = self._read_plot_values(plot)
            sample_values[plot.field_id] = values
            for index, entry in enumerate(plot.series):
                self._histories[plot.field_id][entry.key].append(float(values[index]))

        return time_value, sample_values

    def _history_lists(self) -> dict[str, dict[str, list[float]]]:
        return {
            field_id: {key: list(values) for key, values in series_history.items()}
            for field_id, series_history in self._histories.items()
        }

    def _field_replaces(self) -> tuple[FieldReplace, ...]:
        time_values = np.asarray(list(self._time_history), dtype=np.float32)
        updates: list[FieldReplace] = []
        for plot in PLOTS:
            updates.append(
                FieldReplace(
                    field_id=plot.field_id,
                    values=np.asarray(
                        [list(self._histories[plot.field_id][entry.key]) for entry in plot.series],
                        dtype=np.float32,
                    ),
                    coords={
                        SERIES_DIM: np.asarray([entry.label for entry in plot.series]),
                        TIME_DIM: time_values,
                    },
                )
            )
        return tuple(updates)

    def _field_appends(
        self,
        *,
        time_values: np.ndarray,
        batch_values: dict[str, np.ndarray],
    ) -> tuple[FieldAppend, ...]:
        updates: list[FieldAppend] = []
        for plot in PLOTS:
            updates.append(
                FieldAppend(
                    field_id=plot.field_id,
                    append_dim=TIME_DIM,
                    values=batch_values[plot.field_id],
                    coord_values=time_values,
                    max_length=self.max_samples,
                )
            )
        return tuple(updates)

    def apply_control(self, control_id: str, value) -> bool:
        if control_id == "display_dt":
            try:
                coerced = max(self.dt_ms, float(value))
            except Exception:
                return False
            self.parameters[control_id] = coerced
            self.display_dt_ms = coerced
            return True

        control = CONTROL_BY_ID.get(control_id)
        if control is None and control_id == "downstream_mod_gain":
            control = modulation_gain_control(str(self.runtime.downstream_mod_target))
        if control is None:
            return False

        try:
            coerced = self._coerce_control_value(control, value)
            self.parameters[control_id] = coerced
            if control.target is not None and getattr(self, control.target.owner, None) is not None:
                control.target.write(self, coerced)
            if control_id == "downstream_mod_target":
                new_gain_control = modulation_gain_control(str(coerced))
                new_gain_value = new_gain_control.default_value()
                self.parameters["downstream_mod_gain"] = new_gain_value
                self.runtime.downstream_mod_gain = float(new_gain_value)
                self.emit(
                    ScenePatch(
                        control_updates={
                            "downstream_mod_gain": {
                                "label": new_gain_control.label,
                                "value_spec": new_gain_control.value_spec,
                                "presentation": new_gain_control.presentation,
                            }
                        }
                    )
                )
                self.emit(StatePatch({"downstream_mod_gain": new_gain_value}))
            self._apply_runtime_parameters()
            return True
        except Exception:
            return False


def main() -> None:
    run_app(
        AppSpec(
            session=HHLigandParameterModulationSession,
            title=TITLE,
        )
    )


if __name__ == "__main__":
    main()
