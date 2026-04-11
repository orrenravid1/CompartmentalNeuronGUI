from __future__ import annotations

import math
import time

import numpy as np

from compneurovis import AppSpec, Field, LayoutSpec, LinePlotViewSpec, Scene, run_app
from compneurovis.session import BufferedSession, Error, FieldAppend


class CrashAfterOpenSession(BufferedSession):
    def __init__(
        self,
        *,
        warning_at_update: int = 5,
        crash_after_updates: int = 12,
        update_delay_s: float = 0.15,
    ):
        super().__init__()
        self.warning_at_update = warning_at_update
        self.crash_after_updates = crash_after_updates
        self.update_delay_s = update_delay_s
        self._step = 0
        self._time = 0.0
        self._warning_emitted = False

    def initialize(self) -> Scene:
        field = Field(
            id="demo_trace",
            values=np.array([0.0], dtype=np.float32),
            dims=("time",),
            coords={"time": np.array([0.0], dtype=np.float32)},
        )
        view = LinePlotViewSpec(
            id="trace",
            title="Live trace before failure",
            field_id=field.id,
            x_dim="time",
            x_label="Time",
            y_label="Signal",
            rolling_window=3.0,
        )
        return Scene(
            fields={field.id: field},
            geometries={},
            views={"trace": view},
            layout=LayoutSpec(
                title="Session Error Demo",
                line_plot_view_id="trace",
            ),
        )

    def advance(self) -> None:
        time.sleep(self.update_delay_s)
        self._step += 1
        self._time += 0.1

        if not self._warning_emitted and self._step >= self.warning_at_update:
            self._warning_emitted = True
            self.emit(
                Error(
                    "Intentional nonfatal demo warning from "
                    "CrashAfterOpenSession.advance()."
                )
            )

        if self._step >= self.crash_after_updates:
            raise RuntimeError(
                "Intentional demo failure from CrashAfterOpenSession.advance() "
                "after the window opened."
            )

        value = math.sin(self._time * 4.0)
        self.emit(
            FieldAppend(
                field_id="demo_trace",
                append_dim="time",
                values=np.array([value], dtype=np.float32),
                coord_values=np.array([self._time], dtype=np.float32),
                max_length=200,
            )
        )

    def handle(self, command) -> None:
        del command


run_app(
    AppSpec(
        session=CrashAfterOpenSession,
        title="Session Error Demo",
    )
)
