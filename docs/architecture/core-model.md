---
title: Core Model
summary: Architecture overview — how Field, Geometry, Document, Session, and Frontend fit together.
---

# Core Model

CompNeuroVis is built around five orthogonal primitives that compose cleanly and can be tested independently.

```
Field / Geometry          ← domain data
    ↓
Document                  ← specification (fields + geometries + views + controls + actions + layout)
    ↓
Session (optional)        ← live or replay backend that emits typed updates
    ↓
Frontend                  ← renderer that consumes a Document and applies SessionUpdates
```

## The Primitives

### Field
Dense labeled numpy array with named axes and coordinate metadata. The primary data primitive — see [Field Model](../concepts/field-model.md) for full semantics.

### Geometry
Structural embedding that tells renderers *where* data lives in space. Two kinds:

- `MorphologyGeometry` — one entry per segment: position (n, 3), orientation (n, 3, 3), radius, length, xloc, entity_id, section_name
- `GridGeometry` — regular 2-D grid: two named dims with 1-D coordinate arrays

Geometry holds structural/positional facts. It does not hold time-varying data — that belongs in a `Field`.

### Document
Mutable container for a complete visualization specification:

```python
Document(
    fields={"voltage": voltage_field},
    geometries={"morph": morphology_geom},
    views={"main": MorphologyViewSpec(...)},
    controls={"speed": ControlSpec(...)},
    actions={"reset": ActionSpec(...)},
    layout=LayoutSpec(main_3d_view_id="main", ...),
)
```

`LayoutSpec` controls which views and controls are shown and in what order. If `main_3d_view_id` is `None`, the frontend collapses to a plot-and-controls layout.

`Document` is **mutable** — the frontend modifies `fields` in-place on `FieldReplace` / `FieldAppend` and patches `views`/`controls` on `DocumentPatch`.

### Session
Optional live or replay backend. A `Session`:
1. Returns an initial `Document` from `initialize()` (or `None` if the document is provided externally)
2. Steps forward on each `advance()` call
3. Receives `SessionCommand`s via `handle(command)`
4. Emits `SessionUpdate`s that `read_updates()` drains

Use `BufferedSession` for stateful sessions — it provides an `emit()` method and handles the update queue automatically.

See [Session Protocol](session-protocol.md) for the full message types.

### Frontend
The VisPy/PyQt6 frontend consumes a `Document` and updates panels in response to `SessionUpdate`s. It owns all UI state (selection, slice position, control values). It sends semantic commands to the session — never raw GUI events.

See [VisPy Frontend](vispy-frontend.md) for panel structure and refresh planning.

## Key Invariants

**Field is the data primitive.** Do not introduce `TimeSeries`, `SurfaceData`, or similar types. A `Field` plus a `ViewSpec` is always sufficient.

**Geometry is structural, Field is dynamic.** Positions and connectivity go in `Geometry`. Measured values go in `Field`. Never put time-varying data in `Geometry`.

**Frontends own UI state.** Selection, slice position, and control values live in the frontend's `state` dict. Backends receive `SetControl` / `InvokeAction` — not raw GUI events.

**Prefer the narrowest correct update.** Use `FieldAppend` for incremental live history updates along one dimension. Use `DocumentPatch` for metadata, view property, or control changes that don't require rebuilding the full document. Use `FieldReplace` when replacing a field wholesale. Full replacements are valid, but they are the explicit broader-cost path, not the default.

**`ViewSpec` expresses intent, renderers implement it.** Adding a new visualization means adding a `ViewSpec` subclass and a corresponding renderer/panel — not coupling domain logic into the renderer.
