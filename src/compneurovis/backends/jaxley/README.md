---
title: Jaxley Backend
summary: Jaxley-backed live session and morphology document builder.
---

# Jaxley Backend

`compneurovis.backends.jaxley` provides a Jaxley-native live session with the same high-level shape as the NEURON backend:

- `JaxleySession`: subclass this to build cells, configure the network, and emit live updates
- `JaxleySceneBuilder`: converts Jaxley network compartment geometry into `MorphologyGeometry` plus the default morphology/trace views

By default, the backend splits:

- `segment_display`: latest values for current morphology coloring
- `segment_history`: retained trace history for on-demand trace inspection

The current sampled quantity is voltage by default, but those field ids are role-based rather than voltage-specific. Use `history_capture_mode=HistoryCaptureMode.FULL` when the app needs full all-entity history for retrospective trace selection or playback.
