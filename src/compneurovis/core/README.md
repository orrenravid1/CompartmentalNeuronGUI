---
title: Core Package
summary: Field, geometry, view, control, and scene primitives.
---

# Core Package

`compneurovis.core` defines the stable data and view model:

- `Field`
- `MorphologyGeometry`
- `GridGeometry`
- `Scene`
- `LayoutSpec`
- `View3DHostSpec`
- `ViewSpec`
- `ControlSpec`
- declarative binding helpers such as `AttributeRef` and `SeriesSpec`

`View3DHostSpec` owns 3-D host concerns such as initial camera distance and
starting turntable orientation. Those settings belong here rather than on
`SurfaceViewSpec` or `MorphologyViewSpec`, which describe rendered content.
