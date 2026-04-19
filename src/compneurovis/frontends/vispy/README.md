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

Surface and morphology renderers share one scalar-colormap sampler. Built-in
strings such as `scalar`, `bwr`, `fire`, and `grayscale` still work, custom
ramps can use `ramp:<high>` or `ramp:<low>:<high>`, and optional matplotlib
sampling is available through `mpl:<name>` and `mpl-ramp:<low>:<high>` when
the `matplotlib` extra is installed.

Line-plot panels support both single-trace views and multi-series fields, and the window can mount multiple line-plot views at once while still collapsing cleanly to a 2D-first layout when a scene has no 3D view. Like 3-D views, each line plot now sits inside a small host wrapper so framed chrome and titles stay consistent across panel types. The controls region now uses the same host-wrapper pattern, too, so the whole window presents one consistent panel language.

Line-plot presentation cadence is now frontend-owned. Incoming line-plot
targets mark a plot dirty, and the frontend redraws that plot on a capped
schedule by default instead of assuming every append or state change must force
an immediate pyqtgraph refresh. `LinePlotViewSpec.max_refresh_hz` is the
per-view override seam; values `<= 0` opt out of throttling.
The plot widget itself also enables pyqtgraph clip-to-view and auto
downsampling defaults so redraw cost tracks the visible viewport more closely
when users maximize the window or keep several live traces open.

3-D presentation cadence is now frontend-owned as well. Morphology and surface
refresh targets mark the affected 3-D view dirty, and the frontend presents
those updates on a capped schedule by default instead of repainting the canvas
on every live field update. `MorphologyViewSpec.max_refresh_hz` and
`SurfaceViewSpec.max_refresh_hz` are the per-view override seams; values
`<= 0` opt out of throttling. Both the 3-D and line-plot paths also budget how
many dirty views they present in one flush so one busy live panel does not
starve the rest of the window.

Grid operators such as `GridSliceOperatorSpec` are rendered as host-level
overlays and can also feed other panels such as the line plot without turning
that operator into implicit `SurfaceViewSpec` state.

3D layout is now routed through explicit panel specs:

- `PanelSpec(kind="view_3d")` describes how one or more 3D views are mounted
- 3-D `PanelSpec`s also carry host-level starting camera settings such as
  distance, azimuth, and elevation
- `PanelSpec.operator_ids` selects which operator overlays the host should project
- `IndependentCanvas3DHostPanel` is the current built-in host implementation

That keeps the current one-view-one-canvas behavior intact while leaving room for future shared-canvas or shared-scene hosts.
