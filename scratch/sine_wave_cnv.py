"""
scratch/sine_wave_cnv.py — matplotlib-style module-level API mock.

User-facing code (bottom of this file):

    t_ms = 0.0

    def step():
        global t_ms
        t_ms += DT_MS
        time.sleep(DT_MS / 1000.0)

    cnv.trace("sine", read=lambda: math.sin(...), x=lambda: t_ms)
    cnv.show(step=step)

No threading, no loop, no tick(). cnv.show() hides everything.

Mocked infra (would live in compneurovis/__init__.py + compneurovis/inline.py):
  - module-level trace() / show() functions backed by a singleton _session
  - InlineSession: builds AppSpec, creates window, runs step in daemon thread,
    polls read-lambdas via QTimer at render rate

Run: python scratch/sine_wave_cnv.py
"""
from __future__ import annotations

import math
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PyQt6 import QtCore, QtWidgets
from vispy import app as vispy_app, use

use(app="pyqt6", gl="gl+")

from compneurovis.core.app import (
    AppSpec,
    DataCatalog,
    Field,
    LayoutCatalog,
    LayoutSpec,
    LinePlotViewSpec,
    PanelSpec,
    ViewCatalog,
)
from compneurovis.core.messages import FieldAppend, update_message
from compneurovis.frontends.vispy.frontend import VispyFrontendWindow


# ---------------------------------------------------------------------------
# TraceBinding (same as sine_wave_inline.py — would be compneurovis/bindings/)
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

    def _message(self):
        return update_message(FieldAppend(
            field_id=self._field_id, append_dim="time",
            values=np.array([[self.read()]], dtype=np.float32),
            coord_values=np.array([self.x()], dtype=np.float32),
            max_length=self.max_samples,
        ))


# ---------------------------------------------------------------------------
# InlineSession — the machinery hidden behind cnv.trace() / cnv.show()
# ---------------------------------------------------------------------------

class InlineSession:
    def __init__(self) -> None:
        self._title = "CompNeuroVis"
        self._bindings: list[TraceBinding] = []
        self._qapp: QtWidgets.QApplication | None = None
        self._window: VispyFrontendWindow | None = None
        self._timer: QtCore.QTimer | None = None
        self._stop = threading.Event()

    def _add_trace(self, binding: TraceBinding) -> None:
        binding._register(len(self._bindings))
        self._bindings.append(binding)

    def _build_app_spec(self) -> AppSpec:
        return AppSpec(
            data=DataCatalog(fields={b._field_id: b._initial_field() for b in self._bindings}),
            view_catalog=ViewCatalog(views={b._view_id: b._view_spec() for b in self._bindings}),
            layout_catalog=LayoutCatalog.single(
                LayoutSpec(title=self._title, panels=tuple(b._panel_spec() for b in self._bindings))
            ),
        )

    def _poll(self) -> None:
        """QTimer callback — reads all lambdas, pushes messages, renders."""
        if self._window is None:
            return
        now = time.monotonic()
        for b in self._bindings:
            self._window.handle(b._message())
        self._window.flush_due_refreshes(now=now)

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

        # step runs in a daemon thread — user never sees it
        if step is not None:
            def _run_step():
                while not self._stop.is_set():
                    step()
            threading.Thread(target=_run_step, daemon=True).start()

        self._window.show()
        vispy_app.run()
        self._stop.set()


# ---------------------------------------------------------------------------
# Module-level API (would be compneurovis/__init__.py — like pyplot)
# ---------------------------------------------------------------------------

_session = InlineSession()


def trace(name: str, *, read: Callable[[], float], x: Callable[[], float], **kwargs) -> None:
    _session._add_trace(TraceBinding(name=name, read=read, x=x, **kwargs))


def show(step: Callable[[], None] | None = None, title: str = "CompNeuroVis") -> None:
    _session.show(step=step, title=title)


# ---------------------------------------------------------------------------
# User code — this is all they write
# ---------------------------------------------------------------------------

t_ms = 0.0
FREQ_HZ = 0.5
DT_MS = 16.0


def _step():
    global t_ms
    t_ms += DT_MS
    time.sleep(DT_MS / 1000.0)


trace("sine", read=lambda: math.sin(2 * math.pi * FREQ_HZ * t_ms / 1000.0), x=lambda: t_ms,
      y_min=-1.1, y_max=1.1)

show(step=_step, title="Sine wave")
