---
title: CompNeuroVis Docs
summary: User-facing landing page for the CompNeuroVis documentation site.
---

# CompNeuroVis

CompNeuroVis is a desktop visualization toolkit for computational neuroscience workflows. It is built for the common lab problem of getting simulation output, morphology, surfaces, and linked plots on screen quickly without building a custom GUI from scratch.

## What You Can Do With It

- render a 2-D field as an interactive 3-D surface
- link a surface view to a line-plot slice and controls
- view live NEURON or Jaxley compartment activity on morphology and traces
- replay precomputed frames through the same frontend and layout system

## Start With a Runnable Example

If you want to see the toolkit working before reading architecture docs, start here:

- [Static surface example](getting-started.md#static-surface-first-look)
- [Surface plus cross-section example](getting-started.md#surface-plus-linked-cross-section)
- [Live NEURON example](getting-started.md#live-neuron-morphology-viewer)
- [Live Jaxley example](getting-started.md#live-jaxley-multicell-example)
- [Replay example](getting-started.md#replay-a-precomputed-animation)

## Learn the Toolkit

- [Getting started](getting-started.md) for installation and first-run paths
- [Tutorials](tutorials/build-a-static-surface.md) for adapting examples into your own code
- [Concept docs](concepts/field-model.md) for the stable mental model
- [Architecture docs](architecture/core-model.md) for deeper implementation detail
- [API reference](api/index.md) for generated reference pages over the public authoring surface

## How To Navigate This Site

- Use **Getting Started** if your goal is "show me something working."
- Use **Tutorials** if your goal is "help me build an app like this."
- Use **Concepts** if your goal is "explain the model clearly."
- Use **Architecture** if your goal is "show me how the system is wired."
- Use **API Reference** if your goal is "show me the callable surface and types."
