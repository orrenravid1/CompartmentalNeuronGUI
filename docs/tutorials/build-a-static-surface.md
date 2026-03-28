---
title: Build a Static Surface
summary: Minimal pattern for creating a surface document and running it in the VisPy frontend.
---

# Build a Static Surface

1. Create a `Field` plus `GridGeometry` with `grid_field(...)`.
2. Create a `SurfaceViewSpec` referencing that field and geometry.
3. Call `build_surface_app(...)`.
4. Launch with `run_app(...)`.

See `examples/static_surface_visualizer.py` and `examples/surface_cross_section_visualizer.py`.

