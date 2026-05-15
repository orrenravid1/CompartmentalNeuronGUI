"""
scratch/sine_wave_controls.py — control + action binding proof.

User-facing code (bottom of file):

    freq_hz = [0.5]

    cnv.trace("sine", read=lambda: math.sin(2*math.pi*freq_hz[0]*t_ms/1000.0), x=lambda: t_ms)
    cnv.control("freq_hz", label="Frequency (Hz)",
                get=lambda: freq_hz[0], set=lambda v: freq_hz.__setitem__(0, v),
                min=0.1, max=5.0)
    cnv.action("reset", label="Reset", fn=lambda: t_ms_ref.__setitem__(0, 0.0))
    cnv.show(step=_step, title="Sine with controls")

Proves bidirectional flow:
  forward:  step thread mutates state → QTimer reads lambdas → FieldAppend → window renders
  backward: user moves slider → SetControl emitted → _poll reads window outbound → set() called
            user clicks button  → InvokeAction emitted → _poll reads window outbound → fn() called

Run: python scratch/sine_wave_controls.py
"""
from __future__ import annotations

import math
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
# Binding descriptors
# ---------------------------------------------------------------------------

@dataclass
class TraceBinding:
    name: str
    read: Callable[[], float]
    x: Callable[[], float]
    rolling_window: float = 4000.0
    y_min: float | None = None
    y_max: float | None = None
    max_samples: int = 600
    _field_id: str = field(init=False, default="")
    _view_id: str = field(init=False, default="")

    def _register(self, index: int) -> None:
        self._field_id = f"field_{index}_{self.name}"
        self._view_id  = f"view_{index}_{self.name}"

    def _initial_field(self) -> Field:
        return Field(
            id=self._field_id,
            values=np.array([[self.read()]], dtype=np.float32),
            dims=("series", "time"),
            coords={"series": np.array([self.name]), "time": np.array([self.x()], dtype=np.float32)},
            unit="a.u.",
        )

    def _view_spec(self) -> LinePlotViewSpec:
        return LinePlotViewSpec(
            id=self._view_id, title=self.name, field_id=self._field_id,
            x_dim="time", series_dim="series",
            rolling_window=self.rolling_window, trim_to_rolling_window=True,
            y_min=self.y_min, y_max=self.y_max,
        )

    def _panel_spec(self) -> PanelSpec:
        return PanelSpec(id=f"panel_{self._view_id}", kind="line_plot", view_ids=(self._view_id,))

    def _append_message(self):
        return update_message(FieldAppend(
            field_id=self._field_id, append_dim="time",
            values=np.array([[self.read()]], dtype=np.float32),
            coord_values=np.array([self.x()], dtype=np.float32),
            max_length=self.max_samples,
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
            id=self._control_id,
            label=self.label,
            value_spec=ScalarValueSpec(default=self.get(), min=self.min, max=self.max),
            send_to_backend=True,
        )


@dataclass
class ActionBinding:
    name: str
    label: str
    fn: Callable[[], None]
    _action_id: str = field(init=False, default="")

    def _register(self, index: int) -> None:
        self._action_id = f"action_{index}_{self.name}"

    def _action_spec(self) -> ActionSpec:
        return ActionSpec(id=self._action_id, label=self.label)


# ---------------------------------------------------------------------------
# InlineSession
# ---------------------------------------------------------------------------

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
        b._register(len(self._traces))
        self._traces.append(b)

    def _add_control(self, b: ControlBinding) -> None:
        b._register(len(self._controls))
        self._controls.append(b)

    def _add_action(self, b: ActionBinding) -> None:
        b._register(len(self._actions))
        self._actions.append(b)

    def _build_app_spec(self) -> AppSpec:
        panels = (
            *[t._panel_spec() for t in self._traces],
        )
        if self._controls or self._actions:
            panels = (*panels, PanelSpec(id="panel_controls", kind="controls", view_ids=()))

        return AppSpec(
            data=DataCatalog(fields={t._field_id: t._initial_field() for t in self._traces}),
            view_catalog=ViewCatalog(views={t._view_id: t._view_spec() for t in self._traces}),
            interactions=InteractionCatalog(
                controls={c._control_id: c._control_spec() for c in self._controls},
                actions={a._action_id: a._action_spec() for a in self._actions},
            ),
            layout_catalog=LayoutCatalog.single(
                LayoutSpec(title=self._title, panels=panels)
            ),
        )

    def _poll(self) -> None:
        if self._window is None:
            return
        now = time.monotonic()

        # forward: push trace data to window
        for t in self._traces:
            self._window.handle(t._append_message())
        self._window.flush_due_refreshes(now=now)

        # backward: dispatch window outbound messages to binding callbacks
        for msg in self._window.take_outbound_messages():
            payload = msg.payload
            print(f"[poll] outbound: {type(payload).__name__} {payload}")
            if isinstance(payload, SetControl):
                for c in self._controls:
                    if c._control_id == payload.control_id:
                        print(f"[poll] SetControl matched '{c.name}' → {payload.value}")
                        c.set(payload.value)
                        break
            elif isinstance(payload, InvokeAction):
                for a in self._actions:
                    if a._action_id == payload.action_id:
                        print(f"[poll] InvokeAction matched '{a.name}' → calling fn, t_ms_ref before={t_ms_ref[0]:.1f}")
                        a.fn()
                        print(f"[poll] InvokeAction done, t_ms_ref after={t_ms_ref[0]:.1f}")
                        # x jumped backward — replace all trace fields to clear accumulated data
                        for t in self._traces:
                            self._window.handle(update_message(FieldReplace(
                                field_id=t._field_id,
                                values=np.array([[t.read()]], dtype=np.float32),
                                coords={"series": np.array([t.name]), "time": np.array([t.x()], dtype=np.float32)},
                            )))
                        break
                else:
                    print(f"[poll] InvokeAction NO MATCH: action_id={payload.action_id!r}, known={[a._action_id for a in self._actions]}")

    def show(self, *, step: Callable[[], None] | None = None, title: str = "CompNeuroVis") -> None:
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
            def _run_step():
                while not self._stop.is_set():
                    step()
            threading.Thread(target=_run_step, daemon=True).start()

        self._window.show()
        vispy_app.run()
        self._stop.set()


# ---------------------------------------------------------------------------
# Module-level API
# ---------------------------------------------------------------------------

_session = InlineSession()


def trace(name: str, *, read: Callable[[], float], x: Callable[[], float], **kwargs) -> None:
    _session._add_trace(TraceBinding(name=name, read=read, x=x, **kwargs))


def control(name: str, *, label: str, get: Callable[[], float],
            set: Callable[[Any], None], min: float = 0.0, max: float = 1.0) -> None:
    _session._add_control(ControlBinding(name=name, label=label, get=get, set=set, min=min, max=max))


def action(name: str, *, label: str, fn: Callable[[], None]) -> None:
    _session._add_action(ActionBinding(name=name, label=label, fn=fn))


def show(step: Callable[[], None] | None = None, title: str = "CompNeuroVis") -> None:
    _session.show(step=step, title=title)


# ---------------------------------------------------------------------------
# User code
# ---------------------------------------------------------------------------

freq_hz = [0.5]
t_ms_ref = [0.0]
DT_MS = 16.0


def _step():
    t_ms_ref[0] += DT_MS
    time.sleep(DT_MS / 1000.0)


trace("sine",
      read=lambda: math.sin(2 * math.pi * freq_hz[0] * t_ms_ref[0] / 1000.0),
      x=lambda: t_ms_ref[0],
      y_min=-1.1, y_max=1.1)

control("freq_hz", label="Frequency (Hz)",
        get=lambda: freq_hz[0],
        set=lambda v: freq_hz.__setitem__(0, v),
        min=0.1, max=5.0)

action("reset", label="Reset",
       fn=lambda: t_ms_ref.__setitem__(0, 0.0))

show(step=_step, title="Sine with controls")
