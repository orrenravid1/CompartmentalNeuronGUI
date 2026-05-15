"""
scratch/sine_wave_inline.py — inline/matplotlib-style API mock.

Authoring goal (from the doc, "Capabilities And Parts"):

    vis = cnv.inline()
    vis.enable(cnv.trace("sine", read=lambda: ..., x=lambda: t_ms))
    while vis.is_open():
        t_ms += 16.0
        vis.tick()

No RunSpec, ActorSpec, pipe_transport, or subprocess boilerplate.
User's loop IS the backend. vis.tick() drives data emission + Qt events.

Mocked infra in this file (would live in compneurovis.inline):
  - TraceBinding     — descriptor for a read callback → FieldAppend
  - InlineSession    — builds AppSpec from bindings, owns window, drives tick()

Run: python scratch/sine_wave_inline.py
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PyQt6 import QtWidgets
from vispy import use

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
# Mocked binding descriptors
# (real impl: compneurovis/bindings/trace.py, compneurovis/bindings/control.py)
# ---------------------------------------------------------------------------

@dataclass
class TraceBinding:
    """Backend-side: reads a scalar each tick, emits FieldAppend.
    Frontend-side: auto-declares LinePlotViewSpec in AppSpec.
    """
    name: str
    read: Callable[[], float]           # () -> current value
    x: Callable[[], float]              # () -> current x-axis coord (e.g. time)
    rolling_window: float = 4000.0
    y_min: float | None = None
    y_max: float | None = None
    max_samples: int = 600

    # Internal ids derived at registration time
    _field_id: str = field(init=False, default="")
    _view_id: str = field(init=False, default="")

    def _register(self, index: int) -> None:
        self._field_id = f"inline_field_{index}_{self.name}"
        self._view_id = f"inline_view_{index}_{self.name}"

    def _make_field(self) -> Field:
        return Field(
            id=self._field_id,
            values=np.array([[self.read()]], dtype=np.float32),
            dims=("series", "time"),
            coords={
                "series": np.array([self.name]),
                "time": np.array([self.x()], dtype=np.float32),
            },
            unit="a.u.",
        )

    def _make_view(self) -> LinePlotViewSpec:
        return LinePlotViewSpec(
            id=self._view_id,
            title=self.name,
            field_id=self._field_id,
            x_dim="time",
            series_dim="series",
            rolling_window=self.rolling_window,
            trim_to_rolling_window=True,
            y_min=self.y_min,
            y_max=self.y_max,
        )

    def _make_panel(self) -> PanelSpec:
        return PanelSpec(id=f"panel_{self._view_id}", kind="line_plot", view_ids=(self._view_id,))

    def _make_message(self):
        return update_message(FieldAppend(
            field_id=self._field_id,
            append_dim="time",
            values=np.array([[self.read()]], dtype=np.float32),
            coord_values=np.array([self.x()], dtype=np.float32),
            max_length=self.max_samples,
        ))


# ---------------------------------------------------------------------------
# Mocked InlineSession
# (real impl: compneurovis/inline.py)
# ---------------------------------------------------------------------------

class InlineSession:
    """Drives CompNeuroVis from a user's own loop.

    User code is the backend. No subprocess, no pipe transport.
    tick() emits data → window, then processes Qt events.

    Real impl would:
      - use InProcessTransport (queue) instead of direct window.handle()
      - support ControlBinding (get/set lambdas → widgets)
      - support SelectionBinding
      - lazily start on first tick() instead of requiring explicit start()
    """

    def __init__(self, title: str = "Inline") -> None:
        self._title = title
        self._bindings: list[TraceBinding] = []
        self._window: VispyFrontendWindow | None = None
        self._qapp: QtWidgets.QApplication | None = None
        self._started = False

    def enable(self, binding: TraceBinding) -> "InlineSession":
        binding._register(len(self._bindings))
        self._bindings.append(binding)
        return self

    def _build_app_spec(self) -> AppSpec:
        fields = {b._field_id: b._make_field() for b in self._bindings}
        views  = {b._view_id:  b._make_view()  for b in self._bindings}
        panels = tuple(b._make_panel() for b in self._bindings)
        return AppSpec(
            data=DataCatalog(fields=fields),
            view_catalog=ViewCatalog(views=views),
            layout_catalog=LayoutCatalog.single(
                LayoutSpec(title=self._title, panels=panels)
            ),
        )

    def start(self) -> None:
        if self._started:
            return
        self._started = True

        if QtWidgets.QApplication.instance() is None:
            self._qapp = QtWidgets.QApplication(sys.argv)
        else:
            self._qapp = QtWidgets.QApplication.instance()

        app_spec = self._build_app_spec()
        self._window = VispyFrontendWindow(title=self._title)
        self._window.initialize(app_spec)
        self._window.show()

    def tick(self) -> None:
        """Emit one tick of updates and let Qt process pending events."""
        if not self._started:
            self.start()
        if self._window is None:
            return
        now = time.monotonic()
        for binding in self._bindings:
            self._window.handle(binding._make_message())
        self._window.flush_due_refreshes(now=now)
        QtWidgets.QApplication.processEvents()

    def is_open(self) -> bool:
        if not self._started:
            self.start()
        return self._window is not None and self._window.isVisible()


# ---------------------------------------------------------------------------
# Sugar functions (would live in compneurovis namespace as cnv.trace, cnv.inline)
# ---------------------------------------------------------------------------

def trace(name: str, *, read: Callable[[], float], x: Callable[[], float], **kwargs) -> TraceBinding:
    return TraceBinding(name=name, read=read, x=x, **kwargs)


def inline(title: str = "Inline") -> InlineSession:
    return InlineSession(title=title)


# ---------------------------------------------------------------------------
# Example: sine wave — matplotlib-style authoring
# ---------------------------------------------------------------------------

t_ms = 0.0
FREQ_HZ = 0.5
DT_MS = 16.0

vis = inline(title="Sine wave — inline")
vis.enable(trace(
    "sine",
    read=lambda: math.sin(2 * math.pi * FREQ_HZ * t_ms / 1000.0),
    x=lambda: t_ms,
    y_min=-1.1,
    y_max=1.1,
))

while vis.is_open():
    t_ms += DT_MS
    vis.tick()
    time.sleep(DT_MS / 1000.0)
