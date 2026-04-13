---
title: VisPy Frontend Package
summary: Current PyQt6/VisPy frontend panels, renderers, and window orchestration.
---

# VisPy Frontend Package

This package contains the current runnable frontend:

- `renderers.py`
- `panels.py`
- `frontend.py`

The frontend uses explicit refresh targets and long-lived renderer objects so state changes can update only the affected layers instead of forcing a full scene rebuild.

The line-plot panel supports both single-trace views and multi-series fields, and the window collapses cleanly to a 2D-first layout when a scene has no 3D view.

Grid operators such as `GridSliceOperatorSpec` are rendered as host-level
overlays and can also feed other panels such as the line plot without turning
that operator into implicit `SurfaceViewSpec` state.

3D layout is now routed through explicit host specs:

- `View3DHostSpec` describes how one or more 3D views are mounted
- `View3DHostSpec` also carries host-level starting camera settings such as
  distance, azimuth, and elevation
- `View3DHostSpec.operator_ids` selects which operator overlays the host should project
- `IndependentCanvas3DHostPanel` is the current built-in host implementation

That keeps the current one-view-one-canvas behavior intact while leaving room for future shared-canvas or shared-scene hosts.
