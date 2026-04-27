---
title: VisPy Utilities
summary: Low-level VisPy-only primitives used by frontend renderers.
---

# VisPy Utilities

`compneurovis.frontends.vispy.utils` contains primitives that depend directly on
VisPy and are too low-level to belong in `core`.

Current helpers:

- `CappedCylinderCollection`: batched cylinder-and-cap visual primitive used by
  morphology rendering
