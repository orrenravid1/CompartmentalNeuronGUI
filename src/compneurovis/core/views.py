from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compneurovis.core.state import StateBinding

ValueOrBinding = Any
SelectorValue = Any


@dataclass(frozen=True, slots=True)
class ViewSpec:
    id: str
    title: str = ""


@dataclass(frozen=True, slots=True)
class MorphologyViewSpec(ViewSpec):
    geometry_id: str = "morphology"
    color_field_id: str | None = None
    entity_dim: str = "segment"
    sample_dim: str | None = "time"
    color_map: str = "scalar"
    color_limits: ValueOrBinding = None
    color_norm: str = "auto"
    background_color: ValueOrBinding = "white"


@dataclass(frozen=True, slots=True)
class SurfaceViewSpec(ViewSpec):
    field_id: str = ""
    geometry_id: str | None = None
    color_map: str = "bwr"
    color_limits: tuple[float, float] | None = None
    color_by: str = "height"
    surface_alpha: ValueOrBinding = 1.0
    background_color: ValueOrBinding = "white"
    render_axes: ValueOrBinding = False
    axes_in_middle: ValueOrBinding = True
    tick_count: ValueOrBinding = 5
    tick_length_scale: ValueOrBinding = 1.0
    tick_label_size: ValueOrBinding = 12.0
    axis_label_size: ValueOrBinding = 16.0
    axis_color: ValueOrBinding = "black"
    text_color: ValueOrBinding = "black"
    axis_alpha: ValueOrBinding = 1.0
    axis_labels: tuple[str, str, str] | None = None
    slice_axis_state_key: str | None = None
    slice_position_state_key: str | None = None
    slice_color: ValueOrBinding = "#111111"
    slice_alpha: ValueOrBinding = 0.95
    slice_width: ValueOrBinding = 3.0


@dataclass(frozen=True, slots=True)
class LinePlotViewSpec(ViewSpec):
    field_id: str = ""
    x_dim: str | None = None
    series_dim: str | None = None
    selectors: dict[str, SelectorValue] = field(default_factory=dict)
    orthogonal_slice_state_key: str | None = None
    orthogonal_position_state_key: str | None = None
    x_label: str = "x"
    y_label: str = "y"
    x_unit: str = ""
    y_unit: str = ""
    pen: ValueOrBinding = "k"
    background_color: ValueOrBinding = "w"
    show_legend: bool = True
    series_colors: dict[str, ValueOrBinding] = field(default_factory=dict)
    series_palette: tuple[ValueOrBinding, ...] = ()
    rolling_window: float | None = None
    trim_to_rolling_window: bool = False
    y_min: float | None = None
    y_max: float | None = None
    x_major_tick_spacing: float | None = None
    x_minor_tick_spacing: float | None = None
