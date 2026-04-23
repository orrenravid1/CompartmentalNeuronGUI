"""
Custom LIF backend - pure BufferedSession example of a leaky integrate-and-fire neuron with
threshold/reset spikes, a pulse-injection action, and linked line plots.

Patterns shown:
  - example-owned event-driven solver with no NEURON or Jaxley dependency
  - explicit Field / LinePlotViewSpec / PanelSpec assembly for a live custom backend
  - AttributeRef-backed controls plus action buttons that perturb the model in real time

Run: python examples/custom/lif_backend.py
"""

from __future__ import annotations

from collections import deque
import math
from types import SimpleNamespace

import numpy as np

from compneurovis import (
    ActionSpec,
    AppSpec,
    AttributeRef,
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
from compneurovis.session import BufferedSession, FieldAppend, FieldReplace, InvokeAction, Reset, SetControl, Status


TITLE = "Custom LIF backend"
ROLLING_WINDOW_MS = 500.0
LINE_PLOT_MAX_REFRESH_HZ = 60.0
DEFAULT_DT_MS = 0.25
DEFAULT_DISPLAY_DT_MS = 1.0

TIME_DIM = "time"
SERIES_DIM = "series"

VOLTAGE_FIELD_ID = "lif_voltage_history"
CURRENT_FIELD_ID = "lif_current_history"
EVENT_FIELD_ID = "lif_event_history"

VOLTAGE_VIEW_ID = "lif_voltage_plot"
CURRENT_VIEW_ID = "lif_current_plot"
EVENT_VIEW_ID = "lif_event_plot"

MODEL_DEFAULTS = {
    "rest_voltage_mv": -68.0,
    "reset_voltage_mv": -72.0,
    "threshold_voltage_mv": -50.0,
    "membrane_tau_ms": 18.0,
    "membrane_resistance_mohm": 10.0,
    "tonic_current_na": 1.7,
    "pulse_amplitude_na": 2.8,
    "pulse_decay_ms": 14.0,
    "refractory_ms": 2.5,
}


class LIFModel:
    def __init__(self) -> None:
        self.rest_voltage_mv = float(MODEL_DEFAULTS["rest_voltage_mv"])
        self.reset_voltage_mv = float(MODEL_DEFAULTS["reset_voltage_mv"])
        self.threshold_voltage_mv = float(MODEL_DEFAULTS["threshold_voltage_mv"])
        self.membrane_tau_ms = float(MODEL_DEFAULTS["membrane_tau_ms"])
        self.membrane_resistance_mohm = float(MODEL_DEFAULTS["membrane_resistance_mohm"])
        self.tonic_current_na = float(MODEL_DEFAULTS["tonic_current_na"])
        self.pulse_amplitude_na = float(MODEL_DEFAULTS["pulse_amplitude_na"])
        self.pulse_decay_ms = float(MODEL_DEFAULTS["pulse_decay_ms"])
        self.refractory_ms = float(MODEL_DEFAULTS["refractory_ms"])

        self.reset()

    def reset(self) -> None:
        self.v_mv = float(self.rest_voltage_mv)
        self.pulse_current_na = 0.0
        self.refractory_remaining_ms = 0.0
        self.spike_flag = 0.0

    def deliver_pulse(self, amplitude_na: float | None = None) -> None:
        amplitude = self.pulse_amplitude_na if amplitude_na is None else float(amplitude_na)
        self.pulse_current_na = max(0.0, self.pulse_current_na + amplitude)

    @property
    def total_current_na(self) -> float:
        return float(self.tonic_current_na + self.pulse_current_na)

    @property
    def refractory_fraction(self) -> float:
        if self.refractory_remaining_ms <= 0.0:
            return 0.0
        return float(min(1.0, self.refractory_remaining_ms / max(1e-6, self.refractory_ms)))

    def step(self, dt_ms: float) -> None:
        dt = max(1e-6, float(dt_ms))
        self.spike_flag = 0.0

        decay = dt / max(1e-6, self.pulse_decay_ms)
        self.pulse_current_na = float(max(0.0, self.pulse_current_na * (1.0 - decay)))

        if self.refractory_remaining_ms > 0.0:
            self.refractory_remaining_ms = max(0.0, self.refractory_remaining_ms - dt)
            self.v_mv = float(self.reset_voltage_mv)
            return

        drive_mv = self.membrane_resistance_mohm * self.total_current_na
        dvdt = (self.rest_voltage_mv - self.v_mv + drive_mv) / max(1e-6, self.membrane_tau_ms)
        self.v_mv = float(self.v_mv + dt * dvdt)

        if self.v_mv >= self.threshold_voltage_mv:
            self.spike_flag = 1.0
            self.v_mv = float(self.reset_voltage_mv)
            self.refractory_remaining_ms = max(0.0, float(self.refractory_ms))


VOLTAGE_SERIES = (
    SeriesSpec("vm", "Membrane", source=AttributeRef("model", "v_mv"), color="#00d2be"),
    SeriesSpec("threshold", "Threshold", source=AttributeRef("model", "threshold_voltage_mv"), color="#d1495b"),
    SeriesSpec("reset", "Reset", source=AttributeRef("model", "reset_voltage_mv"), color="#6c757d"),
)

CURRENT_SERIES = (
    SeriesSpec("tonic", "Tonic drive", source=AttributeRef("model", "tonic_current_na"), color="#2356b8"),
    SeriesSpec("pulse", "Pulse drive", source=AttributeRef("model", "pulse_current_na"), color="#ff8c00"),
    SeriesSpec("total", "Total drive", source=AttributeRef("model", "total_current_na"), color="#7d3cff"),
)

EVENT_SERIES = (
    SeriesSpec("spike", "Spike", source=AttributeRef("model", "spike_flag"), color="#ff3366"),
    SeriesSpec("refractory", "Refractory", source=AttributeRef("model", "refractory_fraction"), color="#2f9e44"),
)


def float_control(
    control_id: str,
    label: str,
    default: float,
    min_value: float,
    max_value: float,
    steps: int,
    target: AttributeRef,
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


MODEL_CONTROLS = (
    ControlSpec(
        id="display_dt",
        label="Simulation speed (ms sim/update)",
        value_spec=ScalarValueSpec(
            default=DEFAULT_DISPLAY_DT_MS,
            min=DEFAULT_DT_MS,
            max=20.0,
            value_type="float",
        ),
        presentation=ControlPresentationSpec(kind="slider", steps=199, scale="log"),
        send_to_session=True,
    ),
    float_control(
        "membrane_tau_ms",
        "Membrane tau (ms)",
        MODEL_DEFAULTS["membrane_tau_ms"],
        2.0,
        80.0,
        195,
        AttributeRef("model", "membrane_tau_ms"),
        scale="log",
    ),
    float_control(
        "membrane_resistance_mohm",
        "Resistance (MOhm)",
        MODEL_DEFAULTS["membrane_resistance_mohm"],
        1.0,
        25.0,
        240,
        AttributeRef("model", "membrane_resistance_mohm"),
    ),
    float_control(
        "tonic_current_na",
        "Tonic drive (nA)",
        MODEL_DEFAULTS["tonic_current_na"],
        0.0,
        4.0,
        200,
        AttributeRef("model", "tonic_current_na"),
    ),
    float_control(
        "pulse_amplitude_na",
        "Pulse amplitude (nA)",
        MODEL_DEFAULTS["pulse_amplitude_na"],
        0.0,
        8.0,
        240,
        AttributeRef("model", "pulse_amplitude_na"),
    ),
    float_control(
        "pulse_decay_ms",
        "Pulse decay (ms)",
        MODEL_DEFAULTS["pulse_decay_ms"],
        2.0,
        60.0,
        195,
        AttributeRef("model", "pulse_decay_ms"),
        scale="log",
    ),
    float_control(
        "threshold_voltage_mv",
        "Threshold (mV)",
        MODEL_DEFAULTS["threshold_voltage_mv"],
        -62.0,
        -42.0,
        200,
        AttributeRef("model", "threshold_voltage_mv"),
    ),
    float_control(
        "refractory_ms",
        "Refractory (ms)",
        MODEL_DEFAULTS["refractory_ms"],
        0.0,
        10.0,
        200,
        AttributeRef("model", "refractory_ms"),
    ),
)

MODEL_CONTROL_BY_ID = {control.id: control for control in MODEL_CONTROLS if control.target is not None}

ACTIONS = (
    ActionSpec(id="toggle_pause", label="Pause / Resume", shortcuts=("P",)),
    ActionSpec(id="inject_pulse", label="Inject pulse", shortcuts=("Space",)),
    ActionSpec(id="reset_state", label="Reset state", shortcuts=("R",)),
)


def control_specs(
    *,
    display_dt_ms: float = DEFAULT_DISPLAY_DT_MS,
    dt_ms: float = DEFAULT_DT_MS,
) -> dict[str, ControlSpec]:
    controls: dict[str, ControlSpec] = {}
    for control in MODEL_CONTROLS:
        if control.id == "display_dt":
            controls[control.id] = ControlSpec(
                id=control.id,
                label=control.label,
                value_spec=ScalarValueSpec(
                    default=max(float(dt_ms), float(display_dt_ms)),
                    min=float(dt_ms),
                    max=20.0,
                    value_type="float",
                ),
                presentation=control.presentation,
                send_to_session=control.send_to_session,
                target=control.target,
            )
            continue
        controls[control.id] = control
    return controls


def read_series(root, series: tuple[SeriesSpec, ...]) -> dict[str, float]:
    return {entry.key: float(entry.source.read(root)) for entry in series}


def build_field(
    *,
    field_id: str,
    series: tuple[SeriesSpec, ...],
    time_history: list[float],
    series_history: dict[str, list[float]],
    unit: str,
) -> Field:
    return Field(
        id=field_id,
        values=np.asarray([series_history[entry.key] for entry in series], dtype=np.float32),
        dims=(SERIES_DIM, TIME_DIM),
        coords={
            SERIES_DIM: np.asarray([entry.label for entry in series]),
            TIME_DIM: np.asarray(time_history, dtype=np.float32),
        },
        unit=unit,
    )


def build_scene(
    *,
    time_history: list[float],
    voltage_history: dict[str, list[float]],
    current_history: dict[str, list[float]],
    event_history: dict[str, list[float]],
    controls: dict[str, ControlSpec],
) -> Scene:
    voltage_field = build_field(
        field_id=VOLTAGE_FIELD_ID,
        series=VOLTAGE_SERIES,
        time_history=time_history,
        series_history=voltage_history,
        unit="mV",
    )
    current_field = build_field(
        field_id=CURRENT_FIELD_ID,
        series=CURRENT_SERIES,
        time_history=time_history,
        series_history=current_history,
        unit="nA",
    )
    event_field = build_field(
        field_id=EVENT_FIELD_ID,
        series=EVENT_SERIES,
        time_history=time_history,
        series_history=event_history,
        unit="a.u.",
    )

    views = {
        VOLTAGE_VIEW_ID: LinePlotViewSpec(
            id=VOLTAGE_VIEW_ID,
            title="Membrane voltage",
            field_id=voltage_field.id,
            x_dim=TIME_DIM,
            series_dim=SERIES_DIM,
            x_label="Time",
            x_unit="ms",
            y_label="Voltage",
            y_unit="mV",
            show_legend=True,
            series_colors={entry.label: entry.color for entry in VOLTAGE_SERIES},
            rolling_window=ROLLING_WINDOW_MS,
            trim_to_rolling_window=True,
            y_min=-80.0,
            y_max=-40.0,
            x_major_tick_spacing=100.0,
            x_minor_tick_spacing=25.0,
            max_refresh_hz=LINE_PLOT_MAX_REFRESH_HZ,
        ),
        CURRENT_VIEW_ID: LinePlotViewSpec(
            id=CURRENT_VIEW_ID,
            title="Drive currents",
            field_id=current_field.id,
            x_dim=TIME_DIM,
            series_dim=SERIES_DIM,
            x_label="Time",
            x_unit="ms",
            y_label="Current",
            y_unit="nA",
            show_legend=True,
            series_colors={entry.label: entry.color for entry in CURRENT_SERIES},
            rolling_window=ROLLING_WINDOW_MS,
            trim_to_rolling_window=True,
            y_min=0.0,
            y_max=12.5,
            x_major_tick_spacing=100.0,
            x_minor_tick_spacing=25.0,
            max_refresh_hz=LINE_PLOT_MAX_REFRESH_HZ,
        ),
        EVENT_VIEW_ID: LinePlotViewSpec(
            id=EVENT_VIEW_ID,
            title="Spike events",
            field_id=event_field.id,
            x_dim=TIME_DIM,
            series_dim=SERIES_DIM,
            x_label="Time",
            x_unit="ms",
            y_label="Event state",
            show_legend=True,
            series_colors={entry.label: entry.color for entry in EVENT_SERIES},
            rolling_window=ROLLING_WINDOW_MS,
            trim_to_rolling_window=True,
            y_min=-0.05,
            y_max=1.05,
            x_major_tick_spacing=100.0,
            x_minor_tick_spacing=25.0,
            max_refresh_hz=LINE_PLOT_MAX_REFRESH_HZ,
        ),
    }

    return Scene(
        fields={
            voltage_field.id: voltage_field,
            current_field.id: current_field,
            event_field.id: event_field,
        },
        geometries={},
        views=views,
        controls=controls,
        actions={action.id: action for action in ACTIONS},
        layout=LayoutSpec(
            title=TITLE,
            panels=(
                PanelSpec(id="voltage-panel", kind="line_plot", view_ids=(VOLTAGE_VIEW_ID,)),
                PanelSpec(id="current-panel", kind="line_plot", view_ids=(CURRENT_VIEW_ID,)),
                PanelSpec(id="event-panel", kind="line_plot", view_ids=(EVENT_VIEW_ID,)),
                PanelSpec(
                    id="controls-panel",
                    kind="controls",
                    control_ids=tuple(controls.keys()),
                    action_ids=tuple(action.id for action in ACTIONS),
                ),
            ),
            panel_grid=(("voltage-panel", "current-panel"), ("event-panel", "controls-panel")),
        ),
    )


class CustomLIFSession(BufferedSession):
    @classmethod
    def startup_scene(cls) -> Scene | None:
        model = LIFModel()
        root = SimpleNamespace(model=model)
        time_history = [0.0]
        voltage_history = {key: [value] for key, value in read_series(root, VOLTAGE_SERIES).items()}
        current_history = {key: [value] for key, value in read_series(root, CURRENT_SERIES).items()}
        event_history = {key: [value] for key, value in read_series(root, EVENT_SERIES).items()}
        return build_scene(
            time_history=time_history,
            voltage_history=voltage_history,
            current_history=current_history,
            event_history=event_history,
            controls=control_specs(display_dt_ms=DEFAULT_DISPLAY_DT_MS, dt_ms=DEFAULT_DT_MS),
        )

    def __init__(
        self,
        *,
        dt_ms: float = DEFAULT_DT_MS,
        display_dt_ms: float = DEFAULT_DISPLAY_DT_MS,
        max_samples: int = 2400,
    ) -> None:
        super().__init__()
        self.dt_ms = float(dt_ms)
        self.display_dt_ms = max(self.dt_ms, float(display_dt_ms))
        self.max_samples = int(max_samples)
        self.sim_time_ms = 0.0
        self._paused = False
        self.model = LIFModel()
        self._reset_history()

    def initialize(self) -> Scene:
        self.model.reset()
        self.sim_time_ms = 0.0
        self._paused = False
        self._reset_history()
        self._append_current_sample()
        return build_scene(
            time_history=list(self._time_history),
            voltage_history={key: list(values) for key, values in self._voltage_history.items()},
            current_history={key: list(values) for key, values in self._current_history.items()},
            event_history={key: list(values) for key, values in self._event_history.items()},
            controls=control_specs(display_dt_ms=self.display_dt_ms, dt_ms=self.dt_ms),
        )

    def advance(self) -> None:
        if self._paused:
            return

        time_values: list[float] = []
        voltage_samples = {entry.key: [] for entry in VOLTAGE_SERIES}
        current_samples = {entry.key: [] for entry in CURRENT_SERIES}
        event_samples = {entry.key: [] for entry in EVENT_SERIES}

        for _ in range(self._steps_per_advance()):
            self.model.step(self.dt_ms)
            self.sim_time_ms += self.dt_ms
            time_values.append(self.sim_time_ms)
            self._time_history.append(self.sim_time_ms)

            for entry in VOLTAGE_SERIES:
                value = float(entry.source.read(self))
                voltage_samples[entry.key].append(value)
                self._voltage_history[entry.key].append(value)
            for entry in CURRENT_SERIES:
                value = float(entry.source.read(self))
                current_samples[entry.key].append(value)
                self._current_history[entry.key].append(value)
            for entry in EVENT_SERIES:
                value = float(entry.source.read(self))
                event_samples[entry.key].append(value)
                self._event_history[entry.key].append(value)

        if not time_values:
            return

        time_array = np.asarray(time_values, dtype=np.float32)
        for update in self._field_appends(
            time_values=time_array,
            voltage_samples=voltage_samples,
            current_samples=current_samples,
            event_samples=event_samples,
        ):
            self.emit(update)

    def handle(self, command) -> None:
        if isinstance(command, Reset):
            self._reset_and_replace()
            return
        if isinstance(command, SetControl):
            self.apply_control(command.control_id, command.value)
            return
        if isinstance(command, InvokeAction):
            self.apply_action(command.action_id)

    def idle_sleep(self) -> float:
        return 1.0 / 60.0

    def _steps_per_advance(self) -> int:
        return max(1, int(math.ceil(self.display_dt_ms / self.dt_ms)))

    def _reset_history(self) -> None:
        self._time_history: deque[float] = deque(maxlen=self.max_samples)
        self._voltage_history = {entry.key: deque(maxlen=self.max_samples) for entry in VOLTAGE_SERIES}
        self._current_history = {entry.key: deque(maxlen=self.max_samples) for entry in CURRENT_SERIES}
        self._event_history = {entry.key: deque(maxlen=self.max_samples) for entry in EVENT_SERIES}

    def _append_current_sample(self) -> None:
        self._time_history.append(self.sim_time_ms)
        for entry in VOLTAGE_SERIES:
            self._voltage_history[entry.key].append(float(entry.source.read(self)))
        for entry in CURRENT_SERIES:
            self._current_history[entry.key].append(float(entry.source.read(self)))
        for entry in EVENT_SERIES:
            self._event_history[entry.key].append(float(entry.source.read(self)))

    def _field_appends(
        self,
        *,
        time_values: np.ndarray,
        voltage_samples: dict[str, list[float]],
        current_samples: dict[str, list[float]],
        event_samples: dict[str, list[float]],
    ) -> tuple[FieldAppend, ...]:
        return (
            FieldAppend(
                field_id=VOLTAGE_FIELD_ID,
                append_dim=TIME_DIM,
                values=np.asarray([voltage_samples[entry.key] for entry in VOLTAGE_SERIES], dtype=np.float32),
                coord_values=time_values,
                max_length=self.max_samples,
            ),
            FieldAppend(
                field_id=CURRENT_FIELD_ID,
                append_dim=TIME_DIM,
                values=np.asarray([current_samples[entry.key] for entry in CURRENT_SERIES], dtype=np.float32),
                coord_values=time_values,
                max_length=self.max_samples,
            ),
            FieldAppend(
                field_id=EVENT_FIELD_ID,
                append_dim=TIME_DIM,
                values=np.asarray([event_samples[entry.key] for entry in EVENT_SERIES], dtype=np.float32),
                coord_values=time_values,
                max_length=self.max_samples,
            ),
        )

    def _field_replaces(self) -> tuple[FieldReplace, ...]:
        return (
            FieldReplace(
                field_id=VOLTAGE_FIELD_ID,
                values=np.asarray([self._voltage_history[entry.key] for entry in VOLTAGE_SERIES], dtype=np.float32),
                coords={
                    SERIES_DIM: np.asarray([entry.label for entry in VOLTAGE_SERIES]),
                    TIME_DIM: np.asarray(self._time_history, dtype=np.float32),
                },
            ),
            FieldReplace(
                field_id=CURRENT_FIELD_ID,
                values=np.asarray([self._current_history[entry.key] for entry in CURRENT_SERIES], dtype=np.float32),
                coords={
                    SERIES_DIM: np.asarray([entry.label for entry in CURRENT_SERIES]),
                    TIME_DIM: np.asarray(self._time_history, dtype=np.float32),
                },
            ),
            FieldReplace(
                field_id=EVENT_FIELD_ID,
                values=np.asarray([self._event_history[entry.key] for entry in EVENT_SERIES], dtype=np.float32),
                coords={
                    SERIES_DIM: np.asarray([entry.label for entry in EVENT_SERIES]),
                    TIME_DIM: np.asarray(self._time_history, dtype=np.float32),
                },
            ),
        )

    def _reset_and_replace(self) -> None:
        self.model.reset()
        self.sim_time_ms = 0.0
        self._paused = False
        self._reset_history()
        self._append_current_sample()
        for update in self._field_replaces():
            self.emit(update)
        self.emit(Status("Simulation reset", 1500))

    def apply_control(self, control_id: str, value) -> bool:
        if control_id == "display_dt":
            try:
                self.display_dt_ms = max(self.dt_ms, float(value))
            except Exception:
                return False
            return True

        control = MODEL_CONTROL_BY_ID.get(control_id)
        if control is None or control.target is None:
            return False

        try:
            control.target.write(self, float(value))
        except Exception:
            return False
        return True

    def apply_action(self, action_id: str) -> bool:
        if action_id == "toggle_pause":
            self._paused = not self._paused
            self.emit(Status("Paused" if self._paused else "Running", 1500))
            return True
        if action_id == "inject_pulse":
            self.model.deliver_pulse()
            self.emit(Status(f"Injected pulse: {self.model.pulse_amplitude_na:.2f} nA", 1500))
            return True
        if action_id == "reset_state":
            self._reset_and_replace()
            return True
        return False


run_app(
    AppSpec(
        session=CustomLIFSession,
        title=TITLE,
    )
)
