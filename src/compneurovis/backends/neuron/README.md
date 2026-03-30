---
title: NEURON Backend Package
summary: Live NEURON session and morphology document-building helpers.
---

# NEURON Backend Package

This package contains:

- `NeuronDocumentBuilder`
- `NeuronSession`

By default, the NEURON backend splits latest live display state from retained trace history:

- `voltage_display` for current morphology coloring
- `voltage_trace` for trace history

Use `HistoryCaptureMode.FULL` when the app needs full all-entity history for retrospective trace selection or playback.
