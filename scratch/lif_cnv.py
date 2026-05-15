"""
scratch/lif_cnv.py — LIF example rewritten with the sugar API.

Original: examples/custom/lif_backend.py (~650 lines, explicit AppSpec/BackendBase/RunSpec)
This:     ~120 lines of user code, no framework boilerplate visible

Run: python scratch/lif_cnv.py
"""
from __future__ import annotations

import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from PyQt6 import QtCore, QtWidgets
from vispy import app as vispy_app, use

use(app="pyqt6", gl="gl+")

from compneurovis.core.app import (
    AppSpec,
    DataCatalog,
    Field,
    InteractionCatalog,
    LayoutCatalog,
    LayoutSpec,
    LinePlotViewSpec,
    PanelSpec,
    ViewCatalog,
)
from compneurovis.core.controls import ActionSpec, ControlSpec, ScalarValueSpec
from compneurovis.core.messages import FieldAppend, FieldReplace, InvokeAction, SetControl, update_message
from compneurovis.frontends.vispy.frontend import VispyFrontendWindow


# ---------------------------------------------------------------------------
# Infrastructure (would live in compneurovis/bindings/ + compneurovis/inline.py)
# ---------------------------------------------------------------------------

SeriesReaders = Callable[[], float] | dict[str, Callable[[], float]]


@dataclass
class TraceBinding:
    name: str
    read: SeriesReaders          # single lambda OR {label: lambda} for multi-series
    x: Callable[[], float]
    rolling_window: float = 500.0
    y_min: float | None = None
    y_max: float | None = None
    y_unit: str = "a.u."
    x_unit: str = "ms"
    max_samples: int = 2400
    _field_id: str = field(init=False, default="")
    _view_id: str = field(init=False, default="")
    # Accumulator: step thread writes here, poll thread drains
    _buf_x: list = field(init=False, default_factory=list)
    _buf_vals: list = field(init=False, default_factory=list)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def _register(self, index: int) -> None:
        self._field_id = f"field_{index}_{self.name}"
        self._view_id  = f"view_{index}_{self.name}"

    def _series(self) -> dict[str, Callable[[], float]]:
        if callable(self.read):
            return {self.name: self.read}
        return self.read

    def _sample(self) -> None:
        """Called from step thread after each sim step."""
        series = self._series()
        x = self.x()
        vals = [fn() for fn in series.values()]
        with self._lock:
            self._buf_x.append(x)
            self._buf_vals.append(vals)

    def _drain_message(self):
        """Drain accumulated samples as one batched FieldAppend. Called from poll thread."""
        with self._lock:
            if not self._buf_x:
                return None
            xs = self._buf_x[:]
            vals = self._buf_vals[:]
            self._buf_x.clear()
            self._buf_vals.clear()
        n_series = len(self._series())
        values = np.array(vals, dtype=np.float32).reshape(len(xs), n_series).T  # (n_series, n_time)
        return update_message(FieldAppend(
            field_id=self._field_id, append_dim="time",
            values=values,
            coord_values=np.array(xs, dtype=np.float32),
            max_length=self.max_samples,
        ))

    def _initial_field(self) -> Field:
        series = self._series()
        return Field(
            id=self._field_id,
            values=np.array([[fn()] for fn in series.values()], dtype=np.float32),
            dims=("series", "time"),
            coords={"series": np.array(list(series.keys())), "time": np.array([self.x()], dtype=np.float32)},
            unit=self.y_unit,
        )

    def _view_spec(self) -> LinePlotViewSpec:
        series = self._series()
        return LinePlotViewSpec(
            id=self._view_id, title=self.name, field_id=self._field_id,
            x_dim="time", series_dim="series",
            x_unit=self.x_unit, y_unit=self.y_unit,
            rolling_window=self.rolling_window, trim_to_rolling_window=True,
            y_min=self.y_min, y_max=self.y_max,
            show_legend=len(series) > 1,
            max_refresh_hz=60.0,
        )

    def _panel_spec(self) -> PanelSpec:
        return PanelSpec(id=f"panel_{self._view_id}", kind="line_plot", view_ids=(self._view_id,))

    def _replace_message(self):
        series = self._series()
        values = np.array([[fn()] for fn in series.values()], dtype=np.float32)
        return update_message(FieldReplace(
            field_id=self._field_id,
            values=values,
            coords={"series": np.array(list(series.keys())), "time": np.array([self.x()], dtype=np.float32)},
        ))


@dataclass
class ControlBinding:
    name: str
    label: str
    get: Callable[[], float]
    set: Callable[[Any], None]
    min: float = 0.0
    max: float = 1.0
    _control_id: str = field(init=False, default="")

    def _register(self, index: int) -> None:
        self._control_id = f"ctrl_{index}_{self.name}"

    def _control_spec(self) -> ControlSpec:
        return ControlSpec(
            id=self._control_id, label=self.label,
            value_spec=ScalarValueSpec(default=self.get(), min=self.min, max=self.max),
            send_to_backend=True,
        )


@dataclass
class ActionBinding:
    name: str
    label: str
    fn: Callable[[], None]
    resets_fields: bool = False
    _action_id: str = field(init=False, default="")

    def _register(self, index: int) -> None:
        self._action_id = f"action_{index}_{self.name}"

    def _action_spec(self) -> ActionSpec:
        return ActionSpec(id=self._action_id, label=self.label)


class InlineSession:
    def __init__(self) -> None:
        self._title = "CompNeuroVis"
        self._traces: list[TraceBinding] = []
        self._controls: list[ControlBinding] = []
        self._actions: list[ActionBinding] = []
        self._qapp: QtWidgets.QApplication | None = None
        self._window: VispyFrontendWindow | None = None
        self._timer: QtCore.QTimer | None = None
        self._stop = threading.Event()

    def _add_trace(self, b: TraceBinding) -> None:
        b._register(len(self._traces)); self._traces.append(b)

    def _add_control(self, b: ControlBinding) -> None:
        b._register(len(self._controls)); self._controls.append(b)

    def _add_action(self, b: ActionBinding) -> None:
        b._register(len(self._actions)); self._actions.append(b)

    def _build_app_spec(self) -> AppSpec:
        trace_panels = [t._panel_spec() for t in self._traces]
        ctrl_panel = PanelSpec(
            id="panel_controls", kind="controls",
            control_ids=tuple(c._control_id for c in self._controls),
            action_ids=tuple(a._action_id for a in self._actions),
        ) if self._controls or self._actions else None
        panels = tuple(trace_panels) + ((ctrl_panel,) if ctrl_panel else ())

        # Two-column layout: traces left, controls right (mirrors original lif example)
        n = len(trace_panels)
        if ctrl_panel and n > 0:
            grid = tuple((p.id, ctrl_panel.id) if i == 0 else (p.id,) for i, p in enumerate(trace_panels))
        else:
            grid = None

        layout = LayoutSpec(
            title=self._title,
            panels=panels,
            panel_grid=grid,
        )
        return AppSpec(
            data=DataCatalog(fields={t._field_id: t._initial_field() for t in self._traces}),
            view_catalog=ViewCatalog(views={t._view_id: t._view_spec() for t in self._traces}),
            interactions=InteractionCatalog(
                controls={c._control_id: c._control_spec() for c in self._controls},
                actions={a._action_id: a._action_spec() for a in self._actions},
            ),
            layout_catalog=LayoutCatalog.single(layout),
        )

    def _poll(self) -> None:
        if self._window is None:
            return
        now = time.monotonic()
        for t in self._traces:
            msg = t._drain_message()
            if msg is not None:
                self._window.handle(msg)
        self._window.flush_due_refreshes(now=now)
        for msg in self._window.take_outbound_messages():
            payload = msg.payload
            if isinstance(payload, SetControl):
                for c in self._controls:
                    if c._control_id == payload.control_id:
                        c.set(payload.value)
                        break
            elif isinstance(payload, InvokeAction):
                for a in self._actions:
                    if a._action_id == payload.action_id:
                        a.fn()
                        if a.resets_fields:
                            for t in self._traces:
                                self._window.handle(t._replace_message())
                        break

    def show(self, *, step: Callable[[], None] | None = None,
             dt_ms: float | None = None,
             speed: int | Callable[[], int] | None = None,
             title: str = "CompNeuroVis") -> None:
        self._title = title
        if QtWidgets.QApplication.instance() is None:
            self._qapp = QtWidgets.QApplication(sys.argv)
        else:
            self._qapp = QtWidgets.QApplication.instance()
        signal.signal(signal.SIGINT, lambda *_: self._qapp.quit())
        self._window = VispyFrontendWindow(title=title)
        self._window.initialize(self._build_app_spec())
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._poll)
        self._timer.start(1000 // 60)
        if step is not None:
            _FRAME_MS = 1000.0 / 60.0
            def _run():
                while not self._stop.is_set():
                    frame_start = time.monotonic()
                    n = speed() if callable(speed) else (
                        int(speed) if speed is not None else
                        max(1, int(_FRAME_MS / dt_ms)) if dt_ms else 1
                    )
                    for _ in range(n):
                        if self._stop.is_set():
                            break
                        step()
                        for t in self._traces:
                            t._sample()
                    elapsed_ms = (time.monotonic() - frame_start) * 1000.0
                    sleep_ms = _FRAME_MS - elapsed_ms
                    if sleep_ms > 0.5:
                        time.sleep(sleep_ms / 1000.0)
            threading.Thread(target=_run, daemon=True).start()
        self._window.show()
        vispy_app.run()
        self._stop.set()


_session = InlineSession()

def trace(name: str, *, read: SeriesReaders, x: Callable[[], float], **kwargs) -> None:
    _session._add_trace(TraceBinding(name=name, read=read, x=x, **kwargs))

def control(name: str, *, label: str, get: Callable[[], float],
            set: Callable[[Any], None], min: float = 0.0, max: float = 1.0) -> None:
    _session._add_control(ControlBinding(name=name, label=label, get=get, set=set, min=min, max=max))

def action(name: str, *, label: str, fn: Callable[[], None], resets_fields: bool = False) -> None:
    _session._add_action(ActionBinding(name=name, label=label, fn=fn, resets_fields=resets_fields))

def show(step: Callable[[], None] | None = None,
         dt_ms: float | None = None,
         speed: int | Callable[[], int] | None = None,
         title: str = "CompNeuroVis") -> None:
    _session.show(step=step, dt_ms=dt_ms, speed=speed, title=title)


# ---------------------------------------------------------------------------
# User code — LIF model + app setup
# ---------------------------------------------------------------------------

class LIFModel:
    def __init__(self) -> None:
        self.rest_voltage_mv      = -68.0
        self.reset_voltage_mv     = -72.0
        self.threshold_voltage_mv = -50.0
        self.membrane_tau_ms      = 18.0
        self.membrane_resistance_mohm = 10.0
        self.tonic_current_na     = 1.7
        self.pulse_amplitude_na   = 2.8
        self.pulse_decay_ms       = 14.0
        self.refractory_ms        = 2.5
        self.reset()

    def reset(self) -> None:
        self.v_mv = float(self.rest_voltage_mv)
        self.pulse_current_na = 0.0
        self.refractory_remaining_ms = 0.0
        self.spike_flag = 0.0

    def deliver_pulse(self) -> None:
        self.pulse_current_na = max(0.0, self.pulse_current_na + self.pulse_amplitude_na)

    @property
    def total_current_na(self) -> float:
        return self.tonic_current_na + self.pulse_current_na

    @property
    def refractory_fraction(self) -> float:
        if self.refractory_remaining_ms <= 0.0:
            return 0.0
        return min(1.0, self.refractory_remaining_ms / max(1e-6, self.refractory_ms))

    def step(self, dt_ms: float) -> None:
        self.spike_flag = 0.0
        self.pulse_current_na = max(0.0, self.pulse_current_na * (1.0 - dt_ms / max(1e-6, self.pulse_decay_ms)))
        if self.refractory_remaining_ms > 0.0:
            self.refractory_remaining_ms = max(0.0, self.refractory_remaining_ms - dt_ms)
            self.v_mv = float(self.reset_voltage_mv)
            return
        drive_mv = self.membrane_resistance_mohm * self.total_current_na
        dvdt = (self.rest_voltage_mv - self.v_mv + drive_mv) / max(1e-6, self.membrane_tau_ms)
        self.v_mv += dt_ms * dvdt
        if self.v_mv >= self.threshold_voltage_mv:
            self.spike_flag = 1.0
            self.v_mv = float(self.reset_voltage_mv)
            self.refractory_remaining_ms = float(self.refractory_ms)


DT_MS = 0.25
model = LIFModel()
t_ms = [0.0]
paused = [False]
display_dt_ms = [1.0]


def _step():
    if paused[0]:
        return
    model.step(DT_MS)
    t_ms[0] += DT_MS


def _reset():
    model.reset()
    t_ms[0] = 0.0


trace("Membrane voltage",
      read={"Membrane": lambda: model.v_mv,
            "Threshold": lambda: model.threshold_voltage_mv,
            "Reset V":   lambda: model.reset_voltage_mv},
      x=lambda: t_ms[0], y_min=-80, y_max=-40, y_unit="mV")

trace("Drive currents",
      read={"Tonic": lambda: model.tonic_current_na,
            "Pulse":  lambda: model.pulse_current_na,
            "Total":  lambda: model.total_current_na},
      x=lambda: t_ms[0], y_min=0, y_max=12.5, y_unit="nA")

trace("Spike events",
      read={"Spike":      lambda: model.spike_flag,
            "Refractory": lambda: model.refractory_fraction},
      x=lambda: t_ms[0], y_min=-0.05, y_max=1.05)

control("membrane_tau",  label="Membrane tau (ms)",
        get=lambda: model.membrane_tau_ms,          set=lambda v: setattr(model, "membrane_tau_ms", v),          min=2.0,   max=80.0)
control("resistance",    label="Resistance (MOhm)",
        get=lambda: model.membrane_resistance_mohm, set=lambda v: setattr(model, "membrane_resistance_mohm", v), min=1.0,   max=25.0)
control("tonic_current", label="Tonic drive (nA)",
        get=lambda: model.tonic_current_na,         set=lambda v: setattr(model, "tonic_current_na", v),         min=0.0,   max=4.0)
control("pulse_amp",     label="Pulse amplitude (nA)",
        get=lambda: model.pulse_amplitude_na,       set=lambda v: setattr(model, "pulse_amplitude_na", v),       min=0.0,   max=8.0)
control("pulse_decay",   label="Pulse decay (ms)",
        get=lambda: model.pulse_decay_ms,           set=lambda v: setattr(model, "pulse_decay_ms", v),           min=2.0,   max=60.0)
control("threshold",     label="Threshold (mV)",
        get=lambda: model.threshold_voltage_mv,     set=lambda v: setattr(model, "threshold_voltage_mv", v),     min=-62.0, max=-42.0)
control("refractory",    label="Refractory (ms)",
        get=lambda: model.refractory_ms,            set=lambda v: setattr(model, "refractory_ms", v),            min=0.0,   max=10.0)
control("display_dt",    label="Simulation speed (ms/update)",
        get=lambda: display_dt_ms[0],               set=lambda v: display_dt_ms.__setitem__(0, v),               min=DT_MS, max=20.0)

action("pause", label="Pause / Resume",  fn=lambda: paused.__setitem__(0, not paused[0]))
action("pulse", label="Inject pulse",    fn=lambda: model.deliver_pulse())
action("reset", label="Reset state",     fn=_reset, resets_fields=True)

show(step=_step, dt_ms=DT_MS,
     speed=lambda: max(1, int(display_dt_ms[0] / DT_MS)),
     title="LIF Model")
