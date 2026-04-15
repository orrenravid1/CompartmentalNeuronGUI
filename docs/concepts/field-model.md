---
title: Field Model
summary: How to create, read, and update Field objects — the primary data primitive in CompNeuroVis.
---

# Field Model

`Field` is the primary data primitive. It is a frozen, labeled numpy array with named axes (`dims`) and coordinate metadata (`coords`).

Use this page when you want the mental model for data in CompNeuroVis. If you
are still choosing a workflow, go back to [Getting Started](../getting-started.md)
or [Docs home](../index.md).

## Anatomy of a Field

```python
Field(
    id="voltage",
    values=np.zeros((100, 500)),   # shape must match dims order
    dims=("segment", "time"),
    coords={
        "segment": segment_ids,    # 1-D array, length 100
        "time": t_ms,              # 1-D array, length 500
    },
    unit="mV",   # optional
)
```

Rules enforced at construction:
- `len(dims) == values.ndim`
- `coords.keys() == set(dims)` (exact match, no extras or missing)
- `len(coords[dim]) == values.shape[axis]` for every dim

Fields are **frozen** (`frozen=True`). To change values, use `with_values()`.

## Updating a Field

```python
updated = field.with_values(new_array)                  # replace values only
updated = field.with_values(new_array, coords=new_coords)  # replace values + coords
```

`with_values()` returns a new `Field` with the same `id`, `dims`, and `unit`.

## Selecting from a Field

`field.select(selectors)` returns a new `Field` with one or more dims reduced.

```python
# integer index — removes the dim
field.select({"time": -1})

# float — nearest-coordinate lookup (numeric coords only)
field.select({"time": 150.0})   # picks the closest t_ms value

# string label — coordinate label lookup
field.select({"segment": "soma"})

# slice — keeps the dim, shrinks the coordinate
field.select({"time": slice(100, 200)})

# array of indices — keeps the dim
field.select({"segment": np.array([0, 5, 10])})
```

An integer or string selector that resolves to a single value **removes** that dim from the result. A slice or array selector keeps the dim.

## When to use Field

Use `Field` for any dense measured, simulated, or replayed data:

- membrane voltage by segment by time
- current source density by channel by time
- surface height by y by x
- calcium concentration by compartment by trial

Do not introduce new foundational types (`Timeseries`, `SurfaceData`, etc.) when a `Field` plus a `ViewSpec` is sufficient. "Trace", "surface", and "heatmap" are rendering choices, not data types.

Next steps:

- Read [Geometry Types](geometry-types.md) for the structural side of the data model.
- Read [View and Layout Model](view-layout-model.md) for how fields are consumed by views and visible panels.
