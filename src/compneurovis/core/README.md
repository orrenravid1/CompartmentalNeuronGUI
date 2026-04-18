---
title: Core Package
summary: Field, geometry, view, control, and scene primitives.
---

# Core Package

`compneurovis.core` defines the stable data and view model:

- `Field`
- `MorphologyGeometry`
- `GridGeometry`
- `OperatorSpec`
- `GridSliceOperatorSpec`
- `Scene`
- `LayoutSpec`
- `PanelSpec`
- `ViewSpec`
- `ControlSpec`
- declarative binding helpers such as `AttributeRef` and `SeriesSpec`

`AppSpec` also carries optional `DiagnosticsSpec` settings for app-scoped perf
logging and similar cross-cutting diagnostics.

`PanelSpec` is the current visible-panel seam. A 3-D `PanelSpec` owns host
concerns such as initial camera distance, turntable orientation, and projected
operator overlays. Those settings belong on the visible panel rather than on
`SurfaceViewSpec` or `MorphologyViewSpec`, which describe rendered content.

Operator specs live alongside fields, geometry, and views so derived workflows
such as grid slices can stay reusable across multiple consumers instead of
being baked into one specific view type.
