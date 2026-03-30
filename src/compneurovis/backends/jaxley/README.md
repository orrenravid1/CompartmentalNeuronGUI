---
title: Jaxley Backend
summary: Jaxley-backed live session and morphology document builder.
---

# Jaxley Backend

`compneurovis.backends.jaxley` provides a Jaxley-native live session with the same high-level shape as the NEURON backend:

- `JaxleySession`: subclass this to build cells, configure the network, and emit live updates
- `JaxleyDocumentBuilder`: converts Jaxley network compartment geometry into `MorphologyGeometry` plus the default voltage views

The backend records all compartments by default and streams voltage with `FieldAppend`.
