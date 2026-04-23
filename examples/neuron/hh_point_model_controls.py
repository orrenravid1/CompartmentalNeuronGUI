"""
HH point-model controls - single-compartment NEURON Hodgkin-Huxley example with a live IClamp
amplitude slider plus linked voltage, input-current, and gating plots.

Requires: NEURON
Run: python examples/neuron/hh_point_model_controls.py
"""

from __future__ import annotations

from collections import deque
import math

import numpy as np
from neuron import h

from compneurovis import (
    ActionSpec,
    AppSpec,
    ControlPresentationSpec,
    ControlSpec,
    Field,
    LayoutSpec,
    LinePlotViewSpec,
    PanelSpec,
    ScalarValueSpec,
    Scene,
    run_app,
)
from compneurovis.session import BufferedSession, FieldAppend, FieldReplace, InvokeAction, Reset, SetControl, StatePatch, Status


TITLE = "HH point-model controls"
TIME_DIM = "time"
SERIES_DIM = "series"

VOLTAGE_FIELD_ID = "hh_point_voltage_history"
CURRENT_FIELD_ID = "hh_point_current_history"
STATE_FIELD_ID = "hh_point_state_history"

VOLTAGE_LABEL = "Voltage"
CURRENT_LABEL = "Input current"
STATE_LABELS = ("m", "h", "n")

POINT_LENGTH_UM = 12.6157
POINT_DIAM_UM = 12.6157
CONTINUOUS_CLAMP_DUR_MS = 1.0e9

ROLLING_WINDOW_MS = 80.0
LINE_PLOT_MAX_REFRESH_HZ = 15.0

VOLTAGE_COLOR = "#00d2be"
CURRENT_COLOR = "#2356b8"
STATE_COLORS = {
    "m": "#ff8c00",
    "h": "#ff50b4",
    "n": "#7d3cff",
}

MODEL_CONTROL_IDS = {
    "display_dt",
    "clamp_amp",
    "gnabar",
    "gkbar",
    "gl",
    "el",
    "ena",
    "ek",
    "cm",
    "celsius",
}

DEFAULT_CONTROL_VALUES = {
    "display_dt": 0.1,
    "clamp_amp": 0.0,
    "gnabar": 0.12,
    "gkbar": 0.036,
    "gl": 0.0003,
    "el": -54.3,
    "ena": 50.0,
    "ek": -77.0,
    "cm": 1.0,
    "celsius": 6.3,
}


def float_slider(
    control_id: str,
    label: str,
    default: float,
    min_value: float,
    max_value: float,
    steps: int,
) -> ControlSpec:
    return ControlSpec(
        id=control_id,
        label=label,
        value_spec=ScalarValueSpec(default=default, min=min_value, max=max_value, value_type="float"),
        presentation=ControlPresentationSpec(kind="slider", steps=steps),
        send_to_session=True,
    )


class HHPointModelSession(BufferedSession):
    def __init__(self) -> None:
        super().__init__()
        self.dt = 0.025
        self.display_dt = DEFAULT_CONTROL_VALUES["display_dt"]
        self.v_init = -65.0
        self.max_samples = 4000

        self.clamp_amp = DEFAULT_CONTROL_VALUES["clamp_amp"]
        self.gnabar = DEFAULT_CONTROL_VALUES["gnabar"]
        self.gkbar = DEFAULT_CONTROL_VALUES["gkbar"]
        self.gl = DEFAULT_CONTROL_VALUES["gl"]
        self.el = DEFAULT_CONTROL_VALUES["el"]
        self.ena = DEFAULT_CONTROL_VALUES["ena"]
        self.ek = DEFAULT_CONTROL_VALUES["ek"]
        self.cm = DEFAULT_CONTROL_VALUES["cm"]
        self.celsius = DEFAULT_CONTROL_VALUES["celsius"]

        self.section = None
        self.segment = None
        self.clamp = None

        self._time_history: deque[float] = deque(maxlen=self.max_samples)
        self._voltage_history: deque[float] = deque(maxlen=self.max_samples)
        self._current_history: deque[float] = deque(maxlen=self.max_samples)
        self._state_history: dict[str, deque[float]] = {
            label: deque(maxlen=self.max_samples) for label in STATE_LABELS
        }

    def initialize(self) -> Scene:
        self._build_model()
        self._reset_simulation()
        return self._build_scene()

    def advance(self) -> None:
        time_values: list[float] = []
        voltage_values: list[float] = []
        current_values: list[float] = []
        gate_values: dict[str, list[float]] = {label: [] for label in STATE_LABELS}

        for _ in range(self._steps_per_update()):
            h.fadvance()
            sample = self._sample()
            self._append_sample(sample)

            time_values.append(sample["time"])
            voltage_values.append(sample["voltage"])
            current_values.append(sample["current"])
            for label in STATE_LABELS:
                gate_values[label].append(sample[label])

        if not time_values:
            return

        times = np.asarray(time_values, dtype=np.float32)
        self.emit(
            FieldAppend(
                field_id=VOLTAGE_FIELD_ID,
                append_dim=TIME_DIM,
                values=np.asarray([voltage_values], dtype=np.float32),
                coord_values=times,
                max_length=self.max_samples,
            )
        )
        self.emit(
            FieldAppend(
                field_id=CURRENT_FIELD_ID,
                append_dim=TIME_DIM,
                values=np.asarray([current_values], dtype=np.float32),
                coord_values=times,
                max_length=self.max_samples,
            )
        )
        self.emit(
            FieldAppend(
                field_id=STATE_FIELD_ID,
                append_dim=TIME_DIM,
                values=np.asarray([gate_values[label] for label in STATE_LABELS], dtype=np.float32),
                coord_values=times,
                max_length=self.max_samples,
            )
        )

    def handle(self, command) -> None:
        if isinstance(command, Reset):
            self._reset_and_replace()
            return
        if isinstance(command, SetControl):
            self.apply_control(command.control_id, command.value)
            return
        if isinstance(command, InvokeAction):
            self.apply_action(command.action_id, command.payload)

    def idle_sleep(self) -> float:
        return 1.0 / 60.0

    def control_specs(self) -> dict[str, ControlSpec]:
        return {
            "display_dt": float_slider(
                "display_dt",
                "Visual update interval (ms sim/update)",
                self.display_dt,
                self.dt,
                4.0,
                159,
            ),
            "clamp_amp": float_slider(
                "clamp_amp",
                "IClamp amplitude (nA)",
                self.clamp_amp,
                -0.20,
                0.50,
                280,
            ),
            "gnabar": float_slider(
                "gnabar",
                "Na conductance gNa (S/cm^2)",
                self.gnabar,
                0.01,
                0.30,
                290,
            ),
            "gkbar": float_slider(
                "gkbar",
                "K conductance gK (S/cm^2)",
                self.gkbar,
                0.005,
                0.10,
                190,
            ),
            "gl": float_slider(
                "gl",
                "Leak conductance gL (S/cm^2)",
                self.gl,
                0.00005,
                0.005,
                220,
            ),
            "el": float_slider(
                "el",
                "Leak reversal EL (mV)",
                self.el,
                -80.0,
                -30.0,
                250,
            ),
            "ena": float_slider(
                "ena",
                "Na reversal ENa (mV)",
                self.ena,
                35.0,
                80.0,
                225,
            ),
            "ek": float_slider(
                "ek",
                "K reversal EK (mV)",
                self.ek,
                -110.0,
                -50.0,
                240,
            ),
            "cm": float_slider(
                "cm",
                "Membrane capacitance Cm (uF/cm^2)",
                self.cm,
                0.2,
                3.0,
                280,
            ),
            "celsius": float_slider(
                "celsius",
                "Temperature (degC)",
                self.celsius,
                3.0,
                37.0,
                340,
            ),
        }

    def action_specs(self) -> dict[str, ActionSpec]:
        return {
            "reset": ActionSpec(id="reset", label="Reset", shortcuts=("R",)),
            "reset_defaults": ActionSpec(id="reset_defaults", label="Reset Defaults"),
        }

    def apply_control(self, control_id: str, value) -> bool:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return False

        if control_id not in MODEL_CONTROL_IDS:
            return False

        if control_id == "display_dt":
            self.display_dt = max(self.dt, numeric_value)
        else:
            setattr(self, control_id, numeric_value)
        if self.section is not None and self.segment is not None and self.clamp is not None:
            self._apply_runtime_parameters()
        return True

    def apply_action(self, action_id: str, payload: dict[str, object] | None = None) -> bool:
        del payload
        if action_id != "reset_defaults":
            return False
        self._restore_default_controls()
        self._reset_and_replace()
        self.emit(StatePatch(dict(DEFAULT_CONTROL_VALUES)))
        self.emit(Status("Restored default controls", 1200))
        return True

    def _build_model(self) -> None:
        self.section = h.Section(name="hh_point")
        self.section.L = POINT_LENGTH_UM
        self.section.diam = POINT_DIAM_UM
        self.section.nseg = 1
        self.section.insert("hh")

        self.segment = self.section(0.5)
        self.clamp = h.IClamp(self.segment)
        self._apply_runtime_parameters()

    def _apply_runtime_parameters(self) -> None:
        h.celsius = float(self.celsius)
        self.section.cm = float(self.cm)
        for seg in self.section:
            seg.gnabar_hh = float(self.gnabar)
            seg.gkbar_hh = float(self.gkbar)
            seg.gl_hh = float(self.gl)
            seg.el_hh = float(self.el)
            seg.ena = float(self.ena)
            seg.ek = float(self.ek)

        # Keep the clamp active for the whole run so moving the slider changes
        # the injected current immediately instead of waiting for a pulse window.
        self.clamp.delay = 0.0
        self.clamp.dur = CONTINUOUS_CLAMP_DUR_MS
        self.clamp.amp = float(self.clamp_amp)

    def _sample(self) -> dict[str, float]:
        return {
            "time": float(h.t),
            "voltage": float(self.segment.v),
            "current": float(self.clamp.i),
            "m": float(self.segment.m_hh),
            "h": float(self.segment.h_hh),
            "n": float(self.segment.n_hh),
        }

    def _steps_per_update(self) -> int:
        return max(1, int(math.ceil(self.display_dt / self.dt)))

    def _reset_histories(self) -> None:
        self._time_history.clear()
        self._voltage_history.clear()
        self._current_history.clear()
        for history in self._state_history.values():
            history.clear()

    def _append_sample(self, sample: dict[str, float]) -> None:
        self._time_history.append(sample["time"])
        self._voltage_history.append(sample["voltage"])
        self._current_history.append(sample["current"])
        for label in STATE_LABELS:
            self._state_history[label].append(sample[label])

    def _reset_simulation(self) -> None:
        self._apply_runtime_parameters()
        h.dt = float(self.dt)
        h.finitialize(float(self.v_init))
        self._reset_histories()
        self._append_sample(self._sample())

    def _history_field(
        self,
        *,
        field_id: str,
        labels: tuple[str, ...],
        series_values: list[list[float]],
        unit: str | None = None,
    ) -> Field:
        return Field(
            id=field_id,
            values=np.asarray(series_values, dtype=np.float32),
            dims=(SERIES_DIM, TIME_DIM),
            coords={
                SERIES_DIM: np.asarray(labels),
                TIME_DIM: np.asarray(self._time_history, dtype=np.float32),
            },
            unit=unit,
        )

    def _field_replaces(self) -> tuple[FieldReplace, ...]:
        time_values = np.asarray(self._time_history, dtype=np.float32)
        return (
            FieldReplace(
                field_id=VOLTAGE_FIELD_ID,
                values=np.asarray([list(self._voltage_history)], dtype=np.float32),
                coords={
                    SERIES_DIM: np.asarray([VOLTAGE_LABEL]),
                    TIME_DIM: time_values,
                },
            ),
            FieldReplace(
                field_id=CURRENT_FIELD_ID,
                values=np.asarray([list(self._current_history)], dtype=np.float32),
                coords={
                    SERIES_DIM: np.asarray([CURRENT_LABEL]),
                    TIME_DIM: time_values,
                },
            ),
            FieldReplace(
                field_id=STATE_FIELD_ID,
                values=np.asarray([list(self._state_history[label]) for label in STATE_LABELS], dtype=np.float32),
                coords={
                    SERIES_DIM: np.asarray(STATE_LABELS),
                    TIME_DIM: time_values,
                },
            ),
        )

    def _reset_and_replace(self) -> None:
        self._reset_simulation()
        for update in self._field_replaces():
            self.emit(update)
        self.emit(Status("Simulation reset", 1200))

    def _restore_default_controls(self) -> None:
        self.display_dt = max(self.dt, float(DEFAULT_CONTROL_VALUES["display_dt"]))
        self.clamp_amp = float(DEFAULT_CONTROL_VALUES["clamp_amp"])
        self.gnabar = float(DEFAULT_CONTROL_VALUES["gnabar"])
        self.gkbar = float(DEFAULT_CONTROL_VALUES["gkbar"])
        self.gl = float(DEFAULT_CONTROL_VALUES["gl"])
        self.el = float(DEFAULT_CONTROL_VALUES["el"])
        self.ena = float(DEFAULT_CONTROL_VALUES["ena"])
        self.ek = float(DEFAULT_CONTROL_VALUES["ek"])
        self.cm = float(DEFAULT_CONTROL_VALUES["cm"])
        self.celsius = float(DEFAULT_CONTROL_VALUES["celsius"])

    def _build_scene(self) -> Scene:
        controls = self.control_specs()
        actions = self.action_specs()
        voltage_field = self._history_field(
            field_id=VOLTAGE_FIELD_ID,
            labels=(VOLTAGE_LABEL,),
            series_values=[list(self._voltage_history)],
            unit="mV",
        )
        current_field = self._history_field(
            field_id=CURRENT_FIELD_ID,
            labels=(CURRENT_LABEL,),
            series_values=[list(self._current_history)],
            unit="nA",
        )
        state_field = self._history_field(
            field_id=STATE_FIELD_ID,
            labels=STATE_LABELS,
            series_values=[list(self._state_history[label]) for label in STATE_LABELS],
        )

        return Scene(
            fields={
                voltage_field.id: voltage_field,
                current_field.id: current_field,
                state_field.id: state_field,
            },
            geometries={},
            views={
                "voltage_plot": LinePlotViewSpec(
                    id="voltage_plot",
                    title="Voltage",
                    field_id=VOLTAGE_FIELD_ID,
                    x_dim=TIME_DIM,
                    series_dim=SERIES_DIM,
                    x_label="Time",
                    x_unit="ms",
                    y_label="Voltage",
                    y_unit="mV",
                    show_legend=False,
                    series_colors={VOLTAGE_LABEL: VOLTAGE_COLOR},
                    background_color="white",
                    rolling_window=ROLLING_WINDOW_MS,
                    trim_to_rolling_window=True,
                    max_refresh_hz=LINE_PLOT_MAX_REFRESH_HZ,
                    y_min=-90.0,
                    y_max=60.0,
                    x_major_tick_spacing=20.0,
                    x_minor_tick_spacing=5.0,
                ),
                "current_plot": LinePlotViewSpec(
                    id="current_plot",
                    title="Input current",
                    field_id=CURRENT_FIELD_ID,
                    x_dim=TIME_DIM,
                    series_dim=SERIES_DIM,
                    x_label="Time",
                    x_unit="ms",
                    y_label="Current",
                    y_unit="nA",
                    show_legend=False,
                    series_colors={CURRENT_LABEL: CURRENT_COLOR},
                    background_color="white",
                    rolling_window=ROLLING_WINDOW_MS,
                    trim_to_rolling_window=True,
                    max_refresh_hz=LINE_PLOT_MAX_REFRESH_HZ,
                    y_min=-0.25,
                    y_max=0.55,
                    x_major_tick_spacing=20.0,
                    x_minor_tick_spacing=5.0,
                ),
                "state_plot": LinePlotViewSpec(
                    id="state_plot",
                    title="HH state variables",
                    field_id=STATE_FIELD_ID,
                    x_dim=TIME_DIM,
                    series_dim=SERIES_DIM,
                    x_label="Time",
                    x_unit="ms",
                    y_label="Gate value",
                    show_legend=True,
                    series_colors=STATE_COLORS,
                    background_color="white",
                    rolling_window=ROLLING_WINDOW_MS,
                    trim_to_rolling_window=True,
                    max_refresh_hz=LINE_PLOT_MAX_REFRESH_HZ,
                    y_min=-0.05,
                    y_max=1.05,
                    x_major_tick_spacing=20.0,
                    x_minor_tick_spacing=5.0,
                ),
            },
            controls=controls,
            actions=actions,
            layout=LayoutSpec(
                title=TITLE,
                panels=(
                    PanelSpec(id="voltage-panel", kind="line_plot", view_ids=("voltage_plot",)),
                    PanelSpec(id="current-panel", kind="line_plot", view_ids=("current_plot",)),
                    PanelSpec(id="state-panel", kind="line_plot", view_ids=("state_plot",)),
                    PanelSpec(
                        id="controls-panel",
                        kind="controls",
                        control_ids=tuple(controls.keys()),
                        action_ids=tuple(actions.keys()),
                    ),
                ),
                panel_grid=(
                    ("voltage-panel", "current-panel"),
                    ("state-panel", "controls-panel"),
                ),
            ),
        )


app = AppSpec(session=HHPointModelSession, title=TITLE)
run_app(app)
