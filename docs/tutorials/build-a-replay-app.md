---
title: Build a Replay App
summary: Step-by-step guide to turning saved frames into a runnable CompNeuroVis replay using build_replay_app.
---

# Build a Replay App

Use replay when you already have frames and want the frontend to play them back like a normal app.

Use this tutorial when:

- the data is already computed
- you want the normal CompNeuroVis frontend and layout system
- you do not need a live simulator in the loop

The key public pieces are:

- `ReplaySession`
- `build_replay_app(...)`

You normally use the builder. `ReplaySession` is the session implementation underneath it.

## 1. Build the Scene

Replay still needs a normal `Scene`. The difference is that the data frames already exist.

```python
import numpy as np

from compneurovis import Scene, Field, LayoutSpec, LinePlotViewSpec, PanelSpec

field = Field(
    id="trace",
    values=np.array([0.0], dtype=np.float32),
    dims=("time",),
    coords={"time": np.array([0.0], dtype=np.float32)},
)

scene = Scene(
    fields={field.id: field},
    geometries={},
    views={
        "trace-view": LinePlotViewSpec(
            id="trace-view",
            field_id=field.id,
            x_dim="time",
            x_label="Time",
            y_label="Signal",
        )
    },
    layout=LayoutSpec(
        title="Replay demo",
        panels=(
            PanelSpec(id="trace-panel", kind="line_plot", view_ids=("trace-view",)),
            PanelSpec(id="controls-panel", kind="controls"),
        ),
        panel_grid=(("trace-panel",), ("controls-panel",)),
    ),
)
```

Each framed plot host or controls region is now an explicit `PanelSpec`.
`panel_grid` places those panel ids in row-major order.

## 2. Prepare Frames

`build_replay_app(...)` expects a sequence of frame tuples:

- `values`
- `coords`

Each frame becomes a `FieldReplace` on the target field.

```python
times = np.linspace(0.0, 1.0, 100, dtype=np.float32)
frames = []
for phase in np.linspace(0.0, 2.0 * np.pi, 30, dtype=np.float32):
    values = np.sin(times * 8.0 + phase).astype(np.float32)
    coords = {"time": times}
    frames.append((values, coords))
```

## 3. Build and Run the Replay App

```python
from compneurovis import build_replay_app, run_app

app = build_replay_app(
    scene=scene,
    field_id="trace",
    frames=frames,
)

run_app(app)
```

That is enough to get a replaying line plot.

## What ReplaySession Actually Does

`ReplaySession` is the backend used by `build_replay_app(...)`.

Its job is simple:

- return the provided `Scene` from `initialize()`
- cycle through the frame list in `advance()`
- emit `FieldReplace` for the target field

That makes replay a first-class session mode inside the same architecture as live NEURON or Jaxley apps.

## When To Use Replay

Use replay when:

- the data is already computed
- the app should step through stored frames
- you want the same frontend/layout system without a live simulator in the loop

Use a live backend session instead when the model should be computed during the run.

Next steps:

- Read [Build a static surface](build-a-static-surface.md) if you want to author a non-replay scene from direct field data.
- Read [Session Update Model](../concepts/session-update-model.md) if you want the common session architecture behind replay and live backends.
