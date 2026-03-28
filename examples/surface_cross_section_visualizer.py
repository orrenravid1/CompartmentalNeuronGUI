import numpy as np

from compneurovis import ControlSpec, LinePlotViewSpec, SurfaceViewSpec, build_surface_app, grid_field, run_app


def build_demo_surface():
    x = np.linspace(-4.0, 4.0, 180, dtype=np.float32)
    y = np.linspace(-3.0, 3.0, 160, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    Z = (
        0.9 * np.sin(1.4 * X)
        + 0.35 * X
        + 1.1 * np.cos(0.9 * Y)
        + 0.45 * np.sin(1.8 * Y + 0.5 * X)
        + 0.08 * X * Y
    ).astype(np.float32)
    return x, y, Z


x, y, z = build_demo_surface()
field, geometry = grid_field(
    field_id="cross-section-height",
    values=z,
    x_coords=x,
    y_coords=y,
    x_dim="x",
    y_dim="y",
)

controls = {
    "slice_axis": ControlSpec("slice_axis", "enum", "Slice axis", "x", options=("x", "y")),
    "slice_position": ControlSpec("slice_position", "float", "Slice position", 0.0, min=0.0, max=1.0, steps=200),
}

surface_view = SurfaceViewSpec(
    id="surface",
    title="surface cross-section viewer",
    field_id=field.id,
    geometry_id=geometry.id,
    cmap="fire",
    render_axes=True,
    axes_in_middle=True,
    tick_count=7,
    axis_color="black",
    text_color="black",
    axis_labels=("x", "y", "height"),
    background_color="white",
    surface_alpha=0.9,
    axis_alpha=0.95,
    tick_length_scale=1.0,
    tick_label_size=12.0,
    axis_label_size=16.0,
    slice_axis_state_key="slice_axis",
    slice_position_state_key="slice_position",
)

line_view = LinePlotViewSpec(
    id="cross-section",
    title="Cross section",
    field_id=field.id,
    orthogonal_slice_state_key="slice_axis",
    orthogonal_position_state_key="slice_position",
    y_label="height",
    pen="#1f3c88",
    background_color="white",
)

app = build_surface_app(
    field=field,
    geometry=geometry,
    title="surface cross-section viewer",
    surface_view=surface_view,
    line_view=line_view,
    controls=controls,
)

run_app(app)
