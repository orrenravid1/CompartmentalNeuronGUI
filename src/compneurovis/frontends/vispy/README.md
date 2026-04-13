---
title: VisPy Frontend Package
summary: Current PyQt6/VisPy frontend panels, renderers, and window orchestration.
---

# VisPy Frontend Package

This package contains the current runnable frontend:

- `renderers.py`
- `panels.py`
- `frontend.py`

The frontend uses explicit refresh targets and long-lived renderer objects so state changes can update only the affected layers instead of forcing a full scene rebuild. Surface-axis overlays now split geometry refresh from style refresh and reuse pooled line/text visuals instead of rebuilding every tick label on each control drag.

Line-plot panels support both single-trace views and multi-series fields, and the window can mount multiple line-plot views at once while still collapsing cleanly to a 2D-first layout when a scene has no 3D view. Like 3-D views, each line plot now sits inside a small host wrapper so framed chrome and titles stay consistent across panel types. The controls region now uses the same host-wrapper pattern, too, so the whole window presents one consistent panel language.

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
