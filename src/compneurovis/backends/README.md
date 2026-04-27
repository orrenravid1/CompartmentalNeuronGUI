---
title: Backends Package
summary: Backend-specific sessions and scene builders.
---

# Backends Package

`compneurovis.backends` contains simulator-specific integrations. Current live backends are:

- `neuron`
- `jaxley`

Backend-specific helpers live inside the owning backend package under `utils/`.
Do not add simulator helper roots at the top level of `compneurovis`; keep those
imports local to `compneurovis.backends.<backend>.utils`.
