---
title: Backend Sessions and Utilities
summary: Reference for the NEURON and Jaxley backend authoring surface.
---

# Backend Sessions and Utilities

These modules are useful when you are subclassing a live backend, adjusting the
default scene-building behavior, or using backend-owned morphology helpers.

## NEURON Backend

::: compneurovis.backends.neuron
    options:
      members:
        - NeuronSceneBuilder
        - NeuronSession

## Jaxley Backend

::: compneurovis.backends.jaxley
    options:
      members:
        - JaxleySceneBuilder
        - JaxleySession

## Backend Utilities

Simulator helpers live under the owning backend package:

- `compneurovis.backends.neuron.utils` for NEURON SWC import, JSON round-trip,
  and fallback morphology layout helpers
- `compneurovis.backends.jaxley.utils` for Jaxley SWC loading, cache helpers,
  and geometry translation helpers

Do not use top-level utility roots for simulator helpers.
