"""
Static surface visualizer — renders a 3-D surface from a 2-D height field with interactive axes
and appearance controls. No simulation or session is involved; the surface is computed once at
startup and the UI controls update only the visual properties (colors, transparency, axis style).

Patterns shown:
  - grid_field() to create a Field + GridGeometry from a 2-D array
  - SurfaceViewSpec with axes and colormap
  - StateBinding to wire controls directly to ViewSpec properties without a session

Run: python examples/surface_plot/static_surface_visualizer.py
"""

import numpy as np

from compneurovis import (
    ChoiceValueSpec,
    ControlPresentationSpec,
    ControlSpec,
    PanelSpec,
    ScalarValueSpec,
    StateBinding,
    SurfaceViewSpec,
    build_surface_app,
    grid_field,
    run_app,
)


COLOR_OPTIONS = ("black", "white", "gray", "red", "green", "blue", "orange", "purple")
COLOR_MODES = ("height", "uniform")
SHADING_MODES = ("unlit", "lit")


def choice_control(control_id: str, label: str, default: str, options: tuple[str, ...]) -> ControlSpec:
    return ControlSpec(id=control_id, label=label, value_spec=ChoiceValueSpec(default=default, options=options))


def float_slider(control_id: str, label: str, default: float, min_value: float, max_value: float, steps: int) -> ControlSpec:
    return ControlSpec(
        id=control_id,
        label=label,
        value_spec=ScalarValueSpec(default=default, min=min_value, max=max_value, value_type="float"),
        presentation=ControlPresentationSpec(kind="slider", steps=steps),
    )


def int_control(control_id: str, label: str, default: int, min_value: int, max_value: int) -> ControlSpec:
    return ControlSpec(
        id=control_id,
        label=label,
        value_spec=ScalarValueSpec(default=default, min=min_value, max=max_value, value_type="int"),
    )

# Build a 2-D sinc surface. grid_field() expects values with shape (len(y), len(x)).
x = np.linspace(-3.0, 3.0, 120, dtype=np.float32)
y = np.linspace(-3.0, 3.0, 120, dtype=np.float32)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Z = (np.sinc(R) * 2.0).astype(np.float32)

field, geometry = grid_field(
    field_id="sinc-height",
    values=Z,
    x_coords=x,
    y_coords=y,
    x_dim="x",
    y_dim="y",
)

# Controls define UI widgets. Their ids become state keys that StateBinding references below.
controls = {
    "color_by": choice_control("color_by", "Coloring", "height", COLOR_MODES),
    "surface_color": choice_control("surface_color", "Surface color", "blue", COLOR_OPTIONS),
    "surface_shading": choice_control("surface_shading", "Shading", "unlit", SHADING_MODES),
    "axis_color": choice_control("axis_color", "Axis color", "black", COLOR_OPTIONS),
    "text_color": choice_control("text_color", "Text color", "black", COLOR_OPTIONS),
    "background_color": choice_control("background_color", "Background", "white", COLOR_OPTIONS),
    "surface_alpha": float_slider("surface_alpha", "Surface alpha", 0.9, 0.1, 1.0, 90),
    "tick_count": int_control("tick_count", "Axis ticks", 7, 0, 12),
    "tick_length_scale": float_slider("tick_length_scale", "Tick length", 1.0, 0.0, 3.0, 120),
    "tick_label_size": float_slider("tick_label_size", "Tick text size", 12.0, 6.0, 24.0, 90),
    "axis_label_size": float_slider("axis_label_size", "Axis label size", 16.0, 8.0, 32.0, 96),
    "axis_alpha": float_slider("axis_alpha", "Axes alpha", 1.0, 0.0, 1.0, 100),
}

# StateBinding("key") defers value resolution to the frontend state dict at render time.
# Literal values (e.g. color_map="bwr") are fixed; StateBinding values update live when controls change.
surface_view = SurfaceViewSpec(
    id="surface",
    title="interactive sinc surface",
    field_id=field.id,
    geometry_id=geometry.id,
    color_map="bwr",
    color_by=StateBinding("color_by"),
    surface_color=StateBinding("surface_color"),
    surface_shading=StateBinding("surface_shading"),
    render_axes=True,
    axes_in_middle=True,
    tick_count=StateBinding("tick_count"),
    tick_length_scale=StateBinding("tick_length_scale"),
    tick_label_size=StateBinding("tick_label_size"),
    axis_label_size=StateBinding("axis_label_size"),
    axis_color=StateBinding("axis_color"),
    text_color=StateBinding("text_color"),
    axis_labels=("x", "y", "height"),
    background_color=StateBinding("background_color"),
    surface_alpha=StateBinding("surface_alpha"),
    axis_alpha=StateBinding("axis_alpha"),
)

app = build_surface_app(
    field=field,
    geometry=geometry,
    title="interactive sinc surface",
    surface_view=surface_view,
    controls=controls,
    panels=(
        PanelSpec(id="surface-panel", kind="view_3d", view_ids=("surface",), camera_distance=120.0),
        PanelSpec(id="controls-panel", kind="controls", control_ids=tuple(controls.keys())),
    ),
    panel_grid=(("surface-panel",), ("controls-panel",)),
)

run_app(app)
