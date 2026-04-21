---
title: Tutorials
summary: Step-by-step guides for building specific CompNeuroVis app types from scratch.
---

# Tutorials

Each tutorial walks through one complete app type, explaining every decision along the way. Start with the one that matches your goal.

Prerequisites: run at least one example from [Getting Started](../getting-started.md) first so you have a working install and a reference point.

- [Static Surface](build-a-static-surface.md) — render a 2-D numpy array as an interactive 3-D surface. Start here if you have scalar field data and no simulator.
- [NEURON Session](build-a-neuron-session.md) — live morphology and trace visualization by subclassing `NeuronSession`. Start here for NEURON-backed compartmental models.
- [Jaxley Session](build-a-jaxley-session.md) — same pattern as NEURON, adapted for Jaxley. Start here for JAX-based multi-cell simulations.
- [Replay App](build-a-replay-app.md) — turn saved frames into a playback app via `build_replay_app`. Start here if your simulation already ran and you want to review it.
