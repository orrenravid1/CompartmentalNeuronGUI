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
- `View3DHostSpec`
- `ViewSpec`
- `ControlSpec`
- declarative binding helpers such as `AttributeRef` and `SeriesSpec`

`View3DHostSpec` owns 3-D host concerns such as initial camera distance and
starting turntable orientation. Those settings belong here rather than on
`SurfaceViewSpec` or `MorphologyViewSpec`, which describe rendered content.

Operator specs live alongside fields, geometry, and views so derived workflows
such as grid slices can stay reusable across multiple consumers instead of
being baked into one specific view type.
