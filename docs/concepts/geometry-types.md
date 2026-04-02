---
title: Geometry Types
summary: Mental model for MorphologyGeometry and GridGeometry, and how geometry differs from data.
---

# Geometry Types

Geometry tells the frontend where data lives in space.

The simplest rule in the repo is:

- `Field` answers "what are the values?"
- `Geometry` answers "where do those values live?"

If that split gets blurry, the architecture gets hard to extend.

## MorphologyGeometry

`MorphologyGeometry` is for segment-based biological structure.

Each entry represents one rendered segment and carries:

- position
- orientation
- radius
- length
- entity id
- section name
- x location
- user-facing label

That supports:

- 3D morphology rendering
- picking and selection
- mapping a selected entity id back to a segment label

The important point is that morphology geometry is structural. It is not "voltage geometry" or "calcium geometry." The same geometry can be colored by many different fields over time.

## GridGeometry

`GridGeometry` is for regular 2-D fields, such as surfaces or heatmap-like displays.

It stores:

- the grid dimension names
- the coordinate arrays for each dimension

That supports:

- surface plots
- slice overlays
- line extractions from 2-D fields

Again, the geometry is just the spatial embedding. The actual displayed quantity still lives in a `Field`.

## Geometry Does Not Store Changing Values

Do not put time-varying or measured values into geometry.

Bad pattern:

- storing voltage directly in morphology objects

Good pattern:

- `MorphologyGeometry` for the segment layout
- `Field(id="segment_display", dims=("segment",))` for current display values
- `Field(id="segment_history", dims=("segment", "time"))` for retained history if needed

The same rule applies to surfaces:

- `GridGeometry` for the coordinate lattice
- `Field` for the height, concentration, or analysis output being shown

## Builders Convert Native Structures Into Geometry

Backends often start from simulator-native structures. Document builders convert those into CompNeuroVis geometry.

Examples:

- `NeuronDocumentBuilder` turns NEURON section geometry into `MorphologyGeometry`
- `JaxleyDocumentBuilder` turns Jaxley network compartment geometry into `MorphologyGeometry`

That conversion step is important because it gives the frontend one stable geometry model even when the simulator internals differ.

## Practical Decision Rule

When adding a new visualization:

- if the data lives on segments or compartments, start with `MorphologyGeometry`
- if the data lives on a regular 2-D grid, start with `GridGeometry`
- if neither fits, add a new geometry type only when the spatial model is truly different

Do not add a new geometry type just because the values or the renderer are different.
