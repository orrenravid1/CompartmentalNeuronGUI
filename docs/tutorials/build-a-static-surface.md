---
title: Build a Static Surface
summary: Step-by-step guide to building an interactive surface visualization from a 2-D numpy array.
---

# Build a Static Surface

This tutorial builds the `examples/static_surface_visualizer.py` pattern from scratch.

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

```python continuation
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

```python continuation
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

```python skip
from compneurovis import build_surface_app, run_app

app = build_surface_app(
    field=field,
    geometry=geometry,
    title="sinc surface",
    surface_view=surface_view,
    controls=controls,
)

run_app(app)
```

`build_surface_app()` builds the `Document` and `AppSpec` for you. There is no `Session` — the field values are static.

## Adding a Line Plot Slice

To add a cross-section plot driven by a slider, add `slice_position_state_key` to the `SurfaceViewSpec` and a `LinePlotViewSpec` that references the same field:

```python skip
from compneurovis import LinePlotViewSpec

controls["slice_pos"] = ControlSpec("slice_pos", "float", "Slice Y", 0.0, min=-3.0, max=3.0, steps=120)

surface_view = SurfaceViewSpec(
    ...,
    slice_position_state_key="slice_pos",
    slice_axis_state_key=None,   # fixed to x-axis slice
)

line_view = LinePlotViewSpec(
    id="line",
    field_id=field.id,
    x_dim="x",
    orthogonal_position_state_key="slice_pos",
)
```

Then pass `line_view` to `build_surface_app(line_view=line_view)`.

See `examples/surface_cross_section_visualizer.py` for the full pattern.
