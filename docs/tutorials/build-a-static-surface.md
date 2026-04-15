---
title: Build a Static Surface
summary: Step-by-step guide to building an interactive surface visualization from a 2-D numpy array.
---

# Build a Static Surface

This tutorial builds the `examples/surface_plot/static_surface_visualizer.py`
pattern from scratch. Use it as the default first authoring tutorial when you
want to understand the `Field` + `ViewSpec` + `PanelSpec` model without adding a
live backend.

By the end, you will have:

- one `Field`
- one `GridGeometry`
- one `SurfaceViewSpec`
- optional controls bound through `StateBinding`
- an explicit `LayoutSpec` and `PanelSpec` setup for the visible panels

## 1. Create a Field and Geometry

`grid_field()` is a convenience wrapper that creates both a `Field` and a `GridGeometry` from a 2-D numpy array and coordinate vectors:

```python
import numpy as np
from compneurovis import grid_field

x = np.linspace(-3.0, 3.0, 120, dtype=np.float32)
y = np.linspace(-3.0, 3.0, 120, dtype=np.float32)
X, Y = np.meshgrid(x, y)
Z = (np.sinc(np.sqrt(X**2 + Y**2)) * 2.0).astype(np.float32)

field, geometry = grid_field(
    field_id="sinc-height",
    values=Z,          # shape (len(y), len(x))
    x_coords=x,
    y_coords=y,
    x_dim="x",
    y_dim="y",
)
```

Omit `geometry` from the next steps if your view doesn't need explicit grid coordinates — `SurfaceViewSpec` can infer a unit grid.

## 2. Define Controls (optional)

Controls appear in the right panel and can drive `ViewSpec` properties via `StateBinding`:

```python
from compneurovis import ControlSpec

controls = {
    "surface_alpha": ControlSpec("surface_alpha", "float", "Surface alpha", 0.9, min=0.1, max=1.0, steps=90),
    "background_color": ControlSpec("background_color", "enum", "Background", "white",
                                    options=("black", "white", "gray")),
}
```

Control kinds: `"float"`, `"int"`, `"enum"`.

## 3. Build the ViewSpec

`SurfaceViewSpec` references the field and geometry by id and accepts static values or `StateBinding` placeholders:

```python
from compneurovis import SurfaceViewSpec, StateBinding

surface_view = SurfaceViewSpec(
    id="surface",
    title="sinc surface",
    field_id=field.id,
    geometry_id=geometry.id,
    color_map="fire",
    render_axes=True,
    axes_in_middle=True,
    axis_labels=("x", "y", "height"),
    surface_alpha=StateBinding("surface_alpha"),       # driven by the control above
    background_color=StateBinding("background_color"), # driven by the control above
)
```

Any `ViewSpec` property that accepts a `StateBinding` will resolve to the current control value at render time. Properties without a binding use the literal value you provide.

## 4. Assemble and Run

```python
from compneurovis import PanelSpec, build_surface_app, run_app

app = build_surface_app(
    field=field,
    geometry=geometry,
    title="sinc surface",
    surface_view=surface_view,
    controls=controls,
    panels=(
        PanelSpec(
            id="surface-panel",
            kind="view_3d",
            view_ids=("surface",),
            camera_distance=120.0,
        ),
        PanelSpec(
            id="controls-panel",
            kind="controls",
            control_ids=tuple(controls.keys()),
        ),
    ),
    panel_grid=(("surface-panel",), ("controls-panel",)),
)

run_app(app)
```

`build_surface_app()` builds the `Scene` and `AppSpec` for you. There is no `Session` — the field values are static.
Use a 3-D `PanelSpec` when you want to tune host-level camera settings such as
the initial distance without changing what the `SurfaceViewSpec` renders.

## Adding a Line Plot Slice

To add a cross-section plot driven by a slider, define a `GridSliceOperatorSpec`
and let both the 3-D host and the line plot consume that operator:

```python
from compneurovis import GridSliceOperatorSpec, LinePlotViewSpec

controls["slice_axis"] = ControlSpec("slice_axis", "enum", "Slice axis", "x", options=("x", "y"))
controls["slice_pos"] = ControlSpec("slice_pos", "float", "Slice Y", 0.0, min=-3.0, max=3.0, steps=120)

slice_operator = GridSliceOperatorSpec(
    id="surface-slice",
    field_id=field.id,
    geometry_id=geometry.id,
    axis_state_key="slice_axis",
    position_state_key="slice_pos",
)

line_view = LinePlotViewSpec(
    id="line",
    operator_id=slice_operator.id,
)
```

Then pass `line_views=(line_view,)` and `operators={slice_operator.id: slice_operator}`
to `build_surface_app(...)`, and attach the operator to the 3-D panel through
`PanelSpec.operator_ids`, for example:

`line_views` accepts any number of `LinePlotViewSpec`s. The frontend mounts one
framed plot host per listed view, in the order you pass them.

```python
app = build_surface_app(
    field=field,
    geometry=geometry,
    surface_view=surface_view,
    line_views=(line_view,),
    operators={slice_operator.id: slice_operator},
    controls=controls,
    panels=(
        PanelSpec(
            id="surface-panel",
            kind="view_3d",
            view_ids=("surface",),
            operator_ids=(slice_operator.id,),
        ),
        PanelSpec(
            id="line-panel",
            kind="line_plot",
            view_ids=("line",),
        ),
        PanelSpec(
            id="controls-panel",
            kind="controls",
            control_ids=tuple(controls.keys()),
        ),
    ),
    panel_grid=(("surface-panel", "line-panel"), ("controls-panel",)),
)
```

See `examples/surface_plot/surface_cross_section_visualizer.py` for the full pattern.

Next steps:

- Read [Build a replay app](build-a-replay-app.md) if your data already exists as frames.
- Read [View and Layout Model](../concepts/view-layout-model.md) if you want the composition model behind `ViewSpec`, `PanelSpec`, and `LayoutSpec`.
