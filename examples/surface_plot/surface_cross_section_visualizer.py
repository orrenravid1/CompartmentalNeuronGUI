"""
Surface cross-section visualizer - renders a 3-D height field with a moveable cutting plane and a
linked line plot showing the curve along that slice. Two controls let you choose the slice
axis (x or y) and drag the cutting position; both the surface overlay and the line plot update
together in real time. No simulation or session is involved.

Patterns shown:
  - GridSliceOperatorSpec to model a reusable slice operator over a surface field
  - View3DHostSpec.operator_ids to project operator overlays into the 3-D host without baking them into SurfaceViewSpec
  - LinePlotViewSpec.operator_id to show the operator output as a linked 2-D trace

Run: python examples/surface_plot/surface_cross_section_visualizer.py
"""

import numpy as np

from compneurovis import (
    ControlSpec,
    GridSliceOperatorSpec,
    LinePlotViewSpec,
    SurfaceViewSpec,
    View3DHostSpec,
    build_surface_app,
    grid_field,
    run_app,
)


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
# grid_field returns a (Field, GridGeometry) pair. Both are referenced by id below.
field, geometry = grid_field(
    field_id="cross-section-height",
    values=z,
    x_coords=x,
    y_coords=y,
    x_dim="x",
    y_dim="y",
)

# slice_axis selects which dimension is sliced; slice_position is a 0-1 normalized position.
# Those controls feed the shared operator, not the surface view directly.
controls = {
    "slice_axis": ControlSpec("slice_axis", "enum", "Slice axis", "x", options=("x", "y")),
    "slice_position": ControlSpec("slice_position", "float", "Slice position", 0.0, min=0.0, max=1.0, steps=200),
}

slice_operator = GridSliceOperatorSpec(
    id="surface-slice",
    field_id=field.id,
    geometry_id=geometry.id,
    axis_state_key="slice_axis",
    position_state_key="slice_position",
    fill_alpha=0.16,
)

surface_view = SurfaceViewSpec(
    id="surface",
    title="surface cross-section viewer",
    field_id=field.id,
    geometry_id=geometry.id,
    color_map="bwr",
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
)

# The line plot renders the 1-D output of the shared slice operator.
line_view = LinePlotViewSpec(
    id="cross-section",
    title="Cross section",
    operator_id=slice_operator.id,
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
    operators={slice_operator.id: slice_operator},
    controls=controls,
    view_3d_host=View3DHostSpec(
        id="surface-host",
        view_ids=("surface",),
        operator_ids=(slice_operator.id,),
        camera_distance=30.0,
    ),
)


run_app(app)
