---
title: Builders Package
summary: High-level app builders for common surface, neuron, and replay workflows.
---

# Builders Package

The builder layer keeps common user workflows short:

- `build_neuron_app(...)`
- `build_jaxley_app(...)`
- `build_surface_app(...)`
- `build_replay_app(...)`
- `grid_field(...)`

For live backends, pass a session class or top-level zero-argument factory. Already-created session instances are not the supported path for worker-backed apps.

`build_replay_app(...)` is the replay counterpart. It wraps `ReplaySession` so stored frames can use the same frontend and scene model as live apps.

`build_surface_app(...)` can also assemble grid operators plus linked line
plots over the same field, so surface-specific examples do not need to bake
slice or probe semantics into `SurfaceViewSpec`.
