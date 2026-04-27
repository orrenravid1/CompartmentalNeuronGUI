---
title: API Reference
summary: Generated and curated API reference pages for the public authoring surface of CompNeuroVis.
---

# API Reference

This section is reference material for the public and semi-public authoring
surface of CompNeuroVis.

Do not start here if you are still choosing a workflow or trying to get your
first app running. Start with:

- [Getting Started](../getting-started.md) for first-run paths
- [Tutorials](../tutorials/build-a-static-surface.md) for guided build flows
- [Concepts](../concepts/field-model.md) for the mental model
- [Architecture](../architecture/core-model.md) for implementation boundaries

The generated pages here are intentionally narrower than the full source tree. They focus on the modules humans are expected to use directly:

- the top-level `compneurovis` package
- builder entrypoints
- backend session, scene-builder, and backend-owned utility modules for NEURON
  and Jaxley

Use these pages when you already know roughly what you want and need signatures, types, or hook names.

Good entry pages inside this section:

- [Public API](public-api.md) for top-level exports
- [Builders](builders.md) for high-level app construction helpers
- [Backend Sessions](backends.md) for `NeuronSession`, `JaxleySession`, scene-builder hooks, and backend-owned utility packages
