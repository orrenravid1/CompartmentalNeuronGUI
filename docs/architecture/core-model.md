---
title: Core Model
summary: Architecture overview for the Document, Field, ViewSpec, Session, and Frontend split.
---

# Core Model

CompNeuroVis is organized around a small number of orthogonal primitives:

- `Field`: dense labeled data with named dims and coordinates
- `Geometry`: structural embedding for morphology or regular grids
- `Document`: static fields, geometries, views, controls, and layout
- `Session`: optional live/replay backend that emits typed updates
- `Frontend`: renderer that consumes a `Document` and `SessionUpdate`s

The main design rule is that domain data should be expressed in `Field` and `Geometry`, while rendering choices live in `ViewSpec` classes.

