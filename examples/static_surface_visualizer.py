import numpy as np

from compneurovis import ControlSpec, StateBinding, SurfaceViewSpec, build_surface_app, grid_field, run_app


COLOR_OPTIONS = ("black", "white", "gray", "red", "green", "blue", "orange", "purple")


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

controls = {
    "axis_color": ControlSpec("axis_color", "enum", "Axis color", "black", options=COLOR_OPTIONS),
    "text_color": ControlSpec("text_color", "enum", "Text color", "black", options=COLOR_OPTIONS),
    "background_color": ControlSpec("background_color", "enum", "Background", "white", options=COLOR_OPTIONS),
    "surface_alpha": ControlSpec("surface_alpha", "float", "Surface alpha", 0.9, min=0.1, max=1.0, steps=90),
    "tick_count": ControlSpec("tick_count", "int", "Axis ticks", 7, min=0, max=12),
    "tick_length_scale": ControlSpec("tick_length_scale", "float", "Tick length", 1.0, min=0.0, max=3.0, steps=120),
    "tick_label_size": ControlSpec("tick_label_size", "float", "Tick text size", 12.0, min=6.0, max=24.0, steps=90),
    "axis_label_size": ControlSpec("axis_label_size", "float", "Axis label size", 16.0, min=8.0, max=32.0, steps=96),
    "axis_alpha": ControlSpec("axis_alpha", "float", "Axes alpha", 1.0, min=0.0, max=1.0, steps=100),
}

surface_view = SurfaceViewSpec(
    id="surface",
    title="interactive sinc surface",
    field_id=field.id,
    geometry_id=geometry.id,
    cmap="fire",
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
)


run_app(app)
