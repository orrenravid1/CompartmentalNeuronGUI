"""
Custom FitzHugh-Nagumo backend - complete example of a pure BufferedSession backend with its
own fixed-step RK4 solver, explicit scene assembly, and bound controls/actions. No NEURON or
Jaxley helper is involved.

Patterns shown:
  - model class + custom ODE solver owned entirely by the example
  - explicit Field / LinePlotViewSpec / PanelSpec assembly with no simulator builders
  - AttributeRef and SeriesSpec to bind controls and plotted series onto a nested model
  - BufferedSession startup_scene(), batched FieldAppend updates, and custom actions

Run: python examples/custom/fitzhugh_nagumo_backend.py

Perf logging:
  python examples/custom/fitzhugh_nagumo_backend.py --perf-log
  python examples/custom/fitzhugh_nagumo_backend.py --perf-log --perf-log-stderr
  python examples/custom/fitzhugh_nagumo_backend.py --perf-log-dir .compneurovis/perf-logs/fhn
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from compneurovis._perf import perf_log
from compneurovis import ActionSpec, AppSpec, AttributeRef, ControlSpec, DiagnosticsSpec, Field, LayoutSpec, LinePlotViewSpec, PanelSpec, Scene, SeriesSpec, run_app
from compneurovis.session import BufferedSession, FieldAppend, FieldReplace, InvokeAction, Reset, SetControl, Status


TITLE = "Custom FitzHugh-Nagumo backend"
ROLLING_WINDOW_MS = 600.0

TIME_DIM = "time"
SERIES_DIM = "series"

VOLTAGE_FIELD_ID = "fhn_voltage"
STATE_FIELD_ID = "fhn_state"
TERM_FIELD_ID = "fhn_terms"

VOLTAGE_VIEW_ID = "voltage_plot"
STATE_VIEW_ID = "state_plot"
TERM_VIEW_ID = "term_plot"

LINE_PLOT_MAX_REFRESH_HZ = 30.0


class FitzHughNagumoModel:
    def __init__(self) -> None:
        self.a = 0.7
        self.b = 0.8
        self.tau = 12.5
        self.holding_current = 0.5

        self.exc_weight = 0.9
        self.inh_weight = 0.7
        self.tau_exc = 18.0
        self.tau_inh = 30.0

        self.reset()

    def reset(self, *, v0: float = -1.2, w0: float = -0.62) -> None:
        self.v = float(v0)
        self.w = float(w0)
        self.g_exc = 0.0
        self.g_inh = 0.0

    def deliver_exc_kick(self, weight: float) -> None:
        self.g_exc = max(0.0, self.g_exc + float(weight))

    def deliver_inh_kick(self, weight: float) -> None:
        self.g_inh = max(0.0, self.g_inh + float(weight))

    @property
    def drive_term(self) -> float:
        return float(self.holding_current + self.g_exc - self.g_inh)

    @property
    def cubic_term(self) -> float:
        return float(self.v - (self.v**3) / 3.0)

    @property
    def recovery_term(self) -> float:
        return float(-self.w)

    @property
    def dvdt(self) -> float:
        return float(self.cubic_term + self.recovery_term + self.drive_term)

    def _state_vector(self) -> np.ndarray:
        return np.asarray([self.v, self.w, self.g_exc, self.g_inh], dtype=np.float64)

    def _set_state_vector(self, state: np.ndarray) -> None:
        self.v = float(state[0])
        self.w = float(state[1])
        self.g_exc = float(max(0.0, state[2]))
        self.g_inh = float(max(0.0, state[3]))

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        v, w, g_exc, g_inh = state
        drive = float(self.holding_current) + g_exc - g_inh
        dv = v - (v**3) / 3.0 - w + drive
        dw = (v + float(self.a) - float(self.b) * w) / max(1e-6, float(self.tau))
        dg_exc = -g_exc / max(1e-6, float(self.tau_exc))
        dg_inh = -g_inh / max(1e-6, float(self.tau_inh))
        return np.asarray([dv, dw, dg_exc, dg_inh], dtype=np.float64)

    def _rk4_step(self, state: np.ndarray, dt_ms: float) -> np.ndarray:
        k1 = self._derivatives(state)
        k2 = self._derivatives(state + 0.5 * dt_ms * k1)
        k3 = self._derivatives(state + 0.5 * dt_ms * k2)
        k4 = self._derivatives(state + dt_ms * k3)
        return state + (dt_ms / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def integrate_samples(self, dt_ms: float, sample_count: int) -> np.ndarray:
        dt = max(1e-6, float(dt_ms))
        count = max(1, int(sample_count))
        state = self._state_vector()
        states = np.zeros((count, state.shape[0]), dtype=np.float64)
        for idx in range(count):
            state = self._rk4_step(state, dt)
            states[idx] = state
        self._set_state_vector(states[-1])
        return states


VOLTAGE_SERIES = (
    SeriesSpec("v", "Voltage", source=AttributeRef("model", "v"), color=(0, 210, 190)),
)

STATE_SERIES = (
    SeriesSpec("w", "Recovery", source=AttributeRef("model", "w"), color=(255, 80, 180)),
    SeriesSpec("g_exc", "Exc drive", source=AttributeRef("model", "g_exc"), color=(255, 140, 0)),
    SeriesSpec("g_inh", "Inh drive", source=AttributeRef("model", "g_inh"), color=(160, 0, 255)),
)

TERM_SERIES = (
    SeriesSpec("cubic", "v - v^3/3", source=AttributeRef("model", "cubic_term"), color=(0, 210, 190)),
    SeriesSpec("recovery", "-w", source=AttributeRef("model", "recovery_term"), color=(255, 80, 180)),
    SeriesSpec("drive", "drive", source=AttributeRef("model", "drive_term"), color=(255, 140, 0)),
    SeriesSpec("dvdt", "dV/dt", source=AttributeRef("model", "dvdt"), color=(255, 50, 100)),
)

CONTROLS = (
    ControlSpec("a", "float", "a", 0.7, min=0.1, max=1.5, steps=140, send_to_session=True, target=AttributeRef("model", "a")),
    ControlSpec("b", "float", "b", 0.8, min=0.1, max=1.5, steps=140, send_to_session=True, target=AttributeRef("model", "b")),
    ControlSpec("tau", "float", "tau (ms)", 12.5, min=1.0, max=40.0, steps=195, scale="log", send_to_session=True, target=AttributeRef("model", "tau")),
    ControlSpec(
        "holding_current",
        "float",
        "Holding drive",
        0.5,
        min=-0.5,
        max=1.5,
        steps=200,
        send_to_session=True,
        target=AttributeRef("model", "holding_current"),
    ),
    ControlSpec(
        "exc_weight",
        "float",
        "Exc kick weight",
        0.9,
        min=0.0,
        max=2.5,
        steps=200,
        send_to_session=True,
        target=AttributeRef("model", "exc_weight"),
    ),
    ControlSpec(
        "inh_weight",
        "float",
        "Inh kick weight",
        0.7,
        min=0.0,
        max=2.5,
        steps=200,
        send_to_session=True,
        target=AttributeRef("model", "inh_weight"),
    ),
    ControlSpec(
        "tau_exc",
        "float",
        "Exc decay (ms)",
        18.0,
        min=2.0,
        max=80.0,
        steps=195,
        scale="log",
        send_to_session=True,
        target=AttributeRef("model", "tau_exc"),
    ),
    ControlSpec(
        "tau_inh",
        "float",
        "Inh decay (ms)",
        30.0,
        min=2.0,
        max=80.0,
        steps=195,
        scale="log",
        send_to_session=True,
        target=AttributeRef("model", "tau_inh"),
    ),
)

CONTROL_BY_ID = {control.id: control for control in CONTROLS}

ACTIONS = (
    ActionSpec(id="toggle_pause", label="Pause / Resume", shortcuts=("Space",)),
    ActionSpec(id="deliver_exc_kick", label="Excite", shortcuts=("E",)),
    ActionSpec(id="deliver_inh_kick", label="Inhibit", shortcuts=("Q",)),
    ActionSpec(id="reset", label="Reset", shortcuts=("R",)),
)


def control_specs() -> dict[str, ControlSpec]:
    return {control.id: control for control in CONTROLS}


def read_series(root, series: tuple[SeriesSpec, ...]) -> dict[str, float]:
    return {entry.key: float(entry.source.read(root)) for entry in series}


def build_field(*, field_id: str, series: tuple[SeriesSpec, ...], time_history: list[float], series_history: dict[str, list[float]], unit: str) -> Field:
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
    state_history: dict[str, list[float]],
    term_history: dict[str, list[float]],
    controls: dict[str, ControlSpec],
) -> Scene:
    voltage_field = build_field(
        field_id=VOLTAGE_FIELD_ID,
        series=VOLTAGE_SERIES,
        time_history=time_history,
        series_history=voltage_history,
        unit="a.u.",
    )
    state_field = build_field(
        field_id=STATE_FIELD_ID,
        series=STATE_SERIES,
        time_history=time_history,
        series_history=state_history,
        unit="a.u.",
    )
    term_field = build_field(
        field_id=TERM_FIELD_ID,
        series=TERM_SERIES,
        time_history=time_history,
        series_history=term_history,
        unit="a.u./ms",
    )

    views = {
        VOLTAGE_VIEW_ID: LinePlotViewSpec(
            id=VOLTAGE_VIEW_ID,
            title="Voltage",
            field_id=voltage_field.id,
            x_dim=TIME_DIM,
            series_dim=SERIES_DIM,
            x_label="Time",
            x_unit="ms",
            y_label="V",
            show_legend=False,
            series_colors={entry.label: entry.color for entry in VOLTAGE_SERIES},
            rolling_window=ROLLING_WINDOW_MS,
            trim_to_rolling_window=True,
            y_min=-2.5,
            y_max=2.5,
            x_major_tick_spacing=100.0,
            x_minor_tick_spacing=25.0,
            max_refresh_hz=LINE_PLOT_MAX_REFRESH_HZ
        ),
        STATE_VIEW_ID: LinePlotViewSpec(
            id=STATE_VIEW_ID,
            title="State and synaptic drives",
            field_id=state_field.id,
            x_dim=TIME_DIM,
            series_dim=SERIES_DIM,
            x_label="Time",
            x_unit="ms",
            y_label="State",
            show_legend=True,
            series_colors={entry.label: entry.color for entry in STATE_SERIES},
            rolling_window=ROLLING_WINDOW_MS,
            trim_to_rolling_window=True,
            y_min=-1.5,
            y_max=2.5,
            x_major_tick_spacing=100.0,
            x_minor_tick_spacing=25.0,
            max_refresh_hz=LINE_PLOT_MAX_REFRESH_HZ
        ),
        TERM_VIEW_ID: LinePlotViewSpec(
            id=TERM_VIEW_ID,
            title="Voltage equation terms",
            field_id=term_field.id,
            x_dim=TIME_DIM,
            series_dim=SERIES_DIM,
            x_label="Time",
            x_unit="ms",
            y_label="Contribution",
            show_legend=True,
            series_colors={entry.label: entry.color for entry in TERM_SERIES},
            rolling_window=ROLLING_WINDOW_MS,
            trim_to_rolling_window=True,
            y_min=-3.0,
            y_max=3.0,
            x_major_tick_spacing=100.0,
            x_minor_tick_spacing=25.0,
            max_refresh_hz=LINE_PLOT_MAX_REFRESH_HZ
        ),
    }

    return Scene(
        fields={
            voltage_field.id: voltage_field,
            state_field.id: state_field,
            term_field.id: term_field,
        },
        geometries={},
        views=views,
        controls=controls,
        actions={action.id: action for action in ACTIONS},
        layout=LayoutSpec(
            title=TITLE,
            panels=(
                PanelSpec(id="voltage-panel", kind="line_plot", view_ids=(VOLTAGE_VIEW_ID,)),
                PanelSpec(id="state-panel", kind="line_plot", view_ids=(STATE_VIEW_ID,)),
                PanelSpec(id="terms-panel", kind="line_plot", view_ids=(TERM_VIEW_ID,)),
                PanelSpec(
                    id="controls-panel",
                    kind="controls",
                    control_ids=tuple(control.id for control in CONTROLS),
                    action_ids=tuple(action.id for action in ACTIONS),
                ),
            ),
            panel_grid=(("voltage-panel", "state-panel", "terms-panel"), ("controls-panel",)),
        ),
    )


class CustomFitzHughNagumoSession(BufferedSession):
    @classmethod
    def startup_scene(cls) -> Scene | None:
        model = FitzHughNagumoModel()
        root = SimpleNamespace(model=model)
        time_history = [0.0]
        voltage_history = {key: [value] for key, value in read_series(root, VOLTAGE_SERIES).items()}
        state_history = {key: [value] for key, value in read_series(root, STATE_SERIES).items()}
        term_history = {key: [value] for key, value in read_series(root, TERM_SERIES).items()}
        return build_scene(
            time_history=time_history,
            voltage_history=voltage_history,
            state_history=state_history,
            term_history=term_history,
            controls=control_specs(),
        )

    def __init__(
        self,
        *,
        dt_ms: float = 0.25,
        steps_per_advance: int = 4,
        emit_every_advances: int = 3,
        max_samples: int = 2400,
    ):
        super().__init__()
        self.dt_ms = float(dt_ms)
        self.steps_per_advance = int(steps_per_advance)
        self.emit_every_advances = max(1, int(emit_every_advances))
        self.max_samples = int(max_samples)
        self.sim_time_ms = 0.0
        self._paused = False
        self.model = FitzHughNagumoModel()
        self._reset_history()
        self._reset_pending_emits()

    def initialize(self) -> Scene:
        self.model.reset()
        self.sim_time_ms = 0.0
        self._paused = False
        self._reset_history()
        self._reset_pending_emits()
        self._append_current_sample()
        return build_scene(
            time_history=list(self._time_history),
            voltage_history={key: list(values) for key, values in self._voltage_history.items()},
            state_history={key: list(values) for key, values in self._state_history.items()},
            term_history={key: list(values) for key, values in self._term_history.items()},
            controls=control_specs(),
        )

    def advance(self) -> None:
        if self._paused:
            return

        states = self.model.integrate_samples(self.dt_ms, self.steps_per_advance)
        time_values, voltage_samples, state_samples, term_samples = self._append_states(states)
        self._queue_pending_appends(
            time_values=time_values,
            voltage_samples=voltage_samples,
            state_samples=state_samples,
            term_samples=term_samples,
        )
        self._pending_advance_count += 1
        if self._pending_advance_count >= self.emit_every_advances:
            self._flush_pending_field_appends()

    def handle(self, command) -> None:
        if isinstance(command, Reset):
            self._reset_and_replace()
        elif isinstance(command, SetControl):
            self.apply_control(command.control_id, command.value)
        elif isinstance(command, InvokeAction):
            self.apply_action(command.action_id)

    def idle_sleep(self) -> float:
        return 1 / 30

    def _reset_history(self) -> None:
        self._time_history: deque[float] = deque(maxlen=self.max_samples)
        self._voltage_history = {entry.key: deque(maxlen=self.max_samples) for entry in VOLTAGE_SERIES}
        self._state_history = {entry.key: deque(maxlen=self.max_samples) for entry in STATE_SERIES}
        self._term_history = {entry.key: deque(maxlen=self.max_samples) for entry in TERM_SERIES}

    def _reset_pending_emits(self) -> None:
        self._pending_advance_count = 0
        self._pending_time_values: list[float] = []
        self._pending_voltage_samples = {entry.key: [] for entry in VOLTAGE_SERIES}
        self._pending_state_samples = {entry.key: [] for entry in STATE_SERIES}
        self._pending_term_samples = {entry.key: [] for entry in TERM_SERIES}

    def _append_current_sample(self) -> None:
        self._time_history.append(self.sim_time_ms)
        for entry in VOLTAGE_SERIES:
            self._voltage_history[entry.key].append(float(entry.source.read(self)))
        for entry in STATE_SERIES:
            self._state_history[entry.key].append(float(entry.source.read(self)))
        for entry in TERM_SERIES:
            self._term_history[entry.key].append(float(entry.source.read(self)))

    def _append_states(
        self, states: np.ndarray
    ) -> tuple[np.ndarray, dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
        time_values: list[float] = []
        voltage_samples = {entry.key: [] for entry in VOLTAGE_SERIES}
        state_samples = {entry.key: [] for entry in STATE_SERIES}
        term_samples = {entry.key: [] for entry in TERM_SERIES}

        for state in states:
            self.model._set_state_vector(state)
            self.sim_time_ms += self.dt_ms
            self._time_history.append(self.sim_time_ms)
            time_values.append(self.sim_time_ms)

            for entry in VOLTAGE_SERIES:
                value = float(entry.source.read(self))
                voltage_samples[entry.key].append(value)
                self._voltage_history[entry.key].append(value)
            for entry in STATE_SERIES:
                value = float(entry.source.read(self))
                state_samples[entry.key].append(value)
                self._state_history[entry.key].append(value)
            for entry in TERM_SERIES:
                value = float(entry.source.read(self))
                term_samples[entry.key].append(value)
                self._term_history[entry.key].append(value)

        return np.asarray(time_values, dtype=np.float32), voltage_samples, state_samples, term_samples

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
                field_id=STATE_FIELD_ID,
                values=np.asarray([self._state_history[entry.key] for entry in STATE_SERIES], dtype=np.float32),
                coords={
                    SERIES_DIM: np.asarray([entry.label for entry in STATE_SERIES]),
                    TIME_DIM: np.asarray(self._time_history, dtype=np.float32),
                },
            ),
            FieldReplace(
                field_id=TERM_FIELD_ID,
                values=np.asarray([self._term_history[entry.key] for entry in TERM_SERIES], dtype=np.float32),
                coords={
                    SERIES_DIM: np.asarray([entry.label for entry in TERM_SERIES]),
                    TIME_DIM: np.asarray(self._time_history, dtype=np.float32),
                },
            ),
        )

    def _field_appends(
        self,
        *,
        time_values: np.ndarray,
        voltage_samples: dict[str, list[float]],
        state_samples: dict[str, list[float]],
        term_samples: dict[str, list[float]],
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
                field_id=STATE_FIELD_ID,
                append_dim=TIME_DIM,
                values=np.asarray([state_samples[entry.key] for entry in STATE_SERIES], dtype=np.float32),
                coord_values=time_values,
                max_length=self.max_samples,
            ),
            FieldAppend(
                field_id=TERM_FIELD_ID,
                append_dim=TIME_DIM,
                values=np.asarray([term_samples[entry.key] for entry in TERM_SERIES], dtype=np.float32),
                coord_values=time_values,
                max_length=self.max_samples,
            ),
        )

    def _queue_pending_appends(
        self,
        *,
        time_values: np.ndarray,
        voltage_samples: dict[str, list[float]],
        state_samples: dict[str, list[float]],
        term_samples: dict[str, list[float]],
    ) -> None:
        self._pending_time_values.extend(float(value) for value in time_values)
        for entry in VOLTAGE_SERIES:
            self._pending_voltage_samples[entry.key].extend(voltage_samples[entry.key])
        for entry in STATE_SERIES:
            self._pending_state_samples[entry.key].extend(state_samples[entry.key])
        for entry in TERM_SERIES:
            self._pending_term_samples[entry.key].extend(term_samples[entry.key])

    def _flush_pending_field_appends(self) -> None:
        if not self._pending_time_values:
            return
        time_values = np.asarray(self._pending_time_values, dtype=np.float32)
        voltage_samples = {key: list(values) for key, values in self._pending_voltage_samples.items()}
        state_samples = {key: list(values) for key, values in self._pending_state_samples.items()}
        term_samples = {key: list(values) for key, values in self._pending_term_samples.items()}
        perf_log(
            "custom_backend",
            "flush_appends",
            sample_count=len(self._pending_time_values),
            batched_advances=self._pending_advance_count,
        )
        for update in self._field_appends(
            time_values=time_values,
            voltage_samples=voltage_samples,
            state_samples=state_samples,
            term_samples=term_samples,
        ):
            self.emit(update)
        self._reset_pending_emits()

    def _reset_and_replace(self) -> None:
        self.model.reset()
        self.sim_time_ms = 0.0
        self._paused = False
        self._reset_history()
        self._reset_pending_emits()
        self._append_current_sample()
        perf_log("custom_backend", "reset_and_replace", sim_time_ms=self.sim_time_ms)
        for update in self._field_replaces():
            self.emit(update)
        self.emit(Status("Simulation reset", 1500))

    def apply_control(self, control_id: str, value) -> bool:
        control = CONTROL_BY_ID.get(control_id)
        if control is None or control.target is None:
            return False

        try:
            control.target.write(self, float(value))
        except Exception:
            return False
        perf_log("custom_backend", "apply_control", control_id=control_id, value=float(value))
        return True

    def apply_action(self, action_id: str) -> bool:
        if action_id == "toggle_pause":
            self._paused = not self._paused
            perf_log("custom_backend", "toggle_pause", paused=self._paused)
            self.emit(Status("Paused" if self._paused else "Running", 1500))
            return True
        if action_id == "deliver_exc_kick":
            self.model.deliver_exc_kick(self.model.exc_weight)
            perf_log("custom_backend", "deliver_exc_kick", weight=self.model.exc_weight, g_exc=self.model.g_exc)
            self.emit(Status(f"Excitatory kick: {self.model.exc_weight:.2f}", 1500))
            return True
        if action_id == "deliver_inh_kick":
            self.model.deliver_inh_kick(self.model.inh_weight)
            perf_log("custom_backend", "deliver_inh_kick", weight=self.model.inh_weight, g_inh=self.model.g_inh)
            self.emit(Status(f"Inhibitory kick: {self.model.inh_weight:.2f}", 1500))
            return True
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument(
        "--perf-log",
        action="store_true",
        help="Enable timestamped perf logs under the default .compneurovis/perf-logs directory.",
    )
    parser.add_argument(
        "--perf-log-dir",
        type=Path,
        default=None,
        help="Directory or .jsonl path for perf logs. Implies --perf-log.",
    )
    parser.add_argument(
        "--perf-log-stderr",
        action="store_true",
        help="Also echo perf log records to stderr.",
    )
    return parser.parse_args()


def build_app(
    *,
    perf_log_enabled: bool = False,
    perf_log_dir: Path | None = None,
    perf_log_stderr: bool = False,
) -> AppSpec:
    diagnostics = None
    if perf_log_enabled or perf_log_dir is not None or perf_log_stderr:
        diagnostics = DiagnosticsSpec(
            perf_log_enabled=perf_log_enabled or perf_log_dir is not None,
            perf_log_dir=perf_log_dir,
            perf_echo_stderr=perf_log_stderr,
        )
    return AppSpec(
        session=CustomFitzHughNagumoSession,
        title=TITLE,
        diagnostics=diagnostics,
    )


def main() -> None:
    args = parse_args()
    run_app(
        build_app(
            perf_log_enabled=args.perf_log,
            perf_log_dir=args.perf_log_dir,
            perf_log_stderr=args.perf_log_stderr,
        )
    )


if __name__ == "__main__":
    main()
