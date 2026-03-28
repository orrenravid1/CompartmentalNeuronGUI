---
name: add-static-field-visualization
description: Add or update a static field, surface, or slice-based visualization in CompNeuroVis. Use when building new static documents, grid-backed surfaces, line-plot slices, or field-oriented example apps with the builder layer.
---

# Add a Static Field Visualization

Read `docs/concepts/field-model.md` and `docs/tutorials/build-a-static-surface.md` first.

Prefer this workflow:

- create a `Field`
- create `GridGeometry` only if the view needs explicit grid semantics
- define one or more `ViewSpec` objects
- build the app with `build_surface_app(...)` or a small custom `Document`

Do not introduce new foundational “surface data” or “timeseries data” types when a `Field` plus a view is sufficient.

Update the closest tutorial or example when adding a new visualization pattern.

