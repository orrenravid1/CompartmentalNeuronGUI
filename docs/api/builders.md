---
title: Builders
summary: Generated reference for the primary CompNeuroVis app builders and static-data helpers.
---

# Builders

These functions are the shortest path from domain data or a backend session class to a runnable app.

## Surface and Replay Builders

::: compneurovis.builders.surface
    options:
      members:
        - grid_field
        - build_surface_app

::: compneurovis.builders.replay
    options:
      members:
        - ReplaySession
        - build_replay_app

## Live Backend Builders

::: compneurovis.builders.neuron
    options:
      members:
        - build_neuron_app

::: compneurovis.builders.jaxley
    options:
      members:
        - build_jaxley_app
