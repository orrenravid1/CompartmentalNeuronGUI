"""
Two Line Plots - debug-oriented example that renders two live line plots at once with no 3-D host.

Patterns shown:
  - LayoutSpec.line_plot_view_ids with multiple line plot panels in one window
  - a single appended 2-D Field feeding multiple views through selectors
  - BufferedSession live updates without morphology or surface views

Run: python examples/debug/two_line_plots.py
"""

from __future__ import annotations

import math
import time

import numpy as np

from compneurovis import AppSpec, Field, LayoutSpec, LinePlotViewSpec, Scene, run_app
from compneurovis.session import BufferedSession, FieldAppend


SIGNALS_FIELD_ID = "signals"
SERIES_COORDS = np.array(["fast", "slow"])


def sample_values(time_s: float) -> np.ndarray:
    fast = math.sin(time_s * 6.0) + 0.18 * math.sin(time_s * 15.0 + 0.25)
    slow = 0.7 * math.cos(time_s * 1.6 - 0.2) + 0.22 * math.sin(time_s * 3.4 + 0.5)
    return np.array([[fast], [slow]], dtype=np.float32)


def build_scene() -> Scene:
    field = Field(
        id=SIGNALS_FIELD_ID,
        values=sample_values(0.0),
        dims=("series", "time"),
        coords={
            "series": SERIES_COORDS,
            "time": np.array([0.0], dtype=np.float32),
        },
        unit="a.u.",
    )
    views = {
        "trace-fast": LinePlotViewSpec(
            id="trace-fast",
            title="Fast Trace",
            field_id=field.id,
            x_dim="time",
            selectors={"series": "fast"},
            x_label="Time",
            x_unit="s",
            y_label="Signal",
            pen="#1f4ea8",
            background_color="#fbfcff",
            show_legend=False,
            rolling_window=8.0,
            trim_to_rolling_window=True,
            y_min=-1.4,
            y_max=1.4,
            x_major_tick_spacing=1.0,
        ),
        "trace-slow": LinePlotViewSpec(
            id="trace-slow",
            title="Slow Trace",
            field_id=field.id,
            x_dim="time",
            selectors={"series": "slow"},
            x_label="Time",
            x_unit="s",
            y_label="Signal",
            pen="#b2472f",
            background_color="#fffaf7",
            show_legend=False,
            rolling_window=8.0,
            trim_to_rolling_window=True,
            y_min=-1.4,
            y_max=1.4,
            x_major_tick_spacing=1.0,
        ),
    }
    return Scene(
        fields={field.id: field},
        geometries={},
        views=views,
        layout=LayoutSpec(
            title="Two Line Plots",
            line_plot_view_ids=("trace-fast", "trace-slow"),
        ),
    )


class AnimatedTwoLinePlotsSession(BufferedSession):
    def __init__(self, *, update_delay_s: float = 0.05):
        super().__init__()
        self.update_delay_s = update_delay_s
        self.time_s = 0.0

    def initialize(self) -> Scene:
        return build_scene()

    def advance(self) -> None:
        time.sleep(self.update_delay_s)
        self.time_s += 0.05
        self.emit(
            FieldAppend(
                field_id=SIGNALS_FIELD_ID,
                append_dim="time",
                values=sample_values(self.time_s),
                coord_values=np.array([self.time_s], dtype=np.float32),
                max_length=320,
            )
        )

    def handle(self, command) -> None:
        del command


if __name__ == "__main__":
    run_app(
        AppSpec(
            session=AnimatedTwoLinePlotsSession,
            title="Two Line Plots",
        )
    )
