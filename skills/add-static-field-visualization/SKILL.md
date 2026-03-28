---
name: add-static-field-visualization
description: Add or update a static field, surface, or slice-based visualization in CompNeuroVis. Use when building new static documents, grid-backed surfaces, line-plot slices, or field-oriented example apps with the builder layer.
---

# Add a Static Field Visualization

Read `docs/concepts/field-model.md` and `docs/tutorials/build-a-static-surface.md` first.

Reference implementation: `examples/static_surface_visualizer.py`.

1. Create a `Field` with named dims and coordinate metadata.
2. Create a `GridGeometry` only if the view needs explicit grid semantics; omit it otherwise.
3. Define one or more `ViewSpec` objects that express rendering intent.
4. Assemble the app with `build_surface_app(...)` or a minimal custom `Document` — do not introduce new foundational data types when a `Field` plus a view is sufficient.
5. Update the closest tutorial or example when adding a new visualization pattern.

