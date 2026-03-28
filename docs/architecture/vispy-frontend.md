---
title: VisPy Frontend
summary: Current frontend structure and the responsibilities of panels and renderers.
---

# VisPy Frontend

The current frontend is intentionally simple:

- `Viewport3DPanel` renders morphology or surfaces
- `LinePlotPanel` renders field slices as 2D traces, including multi-series fields
- `ControlsPanel` owns frontend state changes and optional session control dispatch
- `VispyFrontendWindow` coordinates document loading, transport polling, refresh planning, and targeted panel invalidation

Selection is frontend state, not backend state.

Performance-sensitive updates use explicit refresh targets rather than a whole-window redraw. The current invalidation split is:

- morphology refresh
- surface visual refresh
- surface axes refresh
- surface slice overlay refresh
- line-plot refresh
- controls refresh

This keeps common interactions narrow. For example, changing the cross-section slider updates the slice overlay and the derived line plot, but does not rebuild the surface mesh.

If a document omits `main_3d_view_id`, the frontend collapses to a plot-and-controls layout instead of reserving empty space for a hidden viewport.
