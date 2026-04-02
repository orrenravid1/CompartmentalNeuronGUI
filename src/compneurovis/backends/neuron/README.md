---
title: NEURON Backend Package
summary: Live NEURON session and morphology document-building helpers.
---

# NEURON Backend Package

This package contains:

- `NeuronDocumentBuilder`
- `NeuronSession`

By default, the NEURON backend splits latest live display state from retained trace history:

- `segment_display` for current morphology coloring
- `segment_history` for trace history

The current sampled quantity is voltage by default, but those field ids are role-based rather than voltage-specific. Use `HistoryCaptureMode.FULL` when the app needs full all-entity history for retrospective trace selection or playback.
