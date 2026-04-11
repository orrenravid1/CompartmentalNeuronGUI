---
title: Core Model
summary: Architecture overview - how Field, Geometry, Scene, Session, and Frontend fit together.
---

# Core Model

CompNeuroVis is built around five orthogonal primitives that compose cleanly and can be tested independently.

```
Field / Geometry          <- domain data
    |
Scene                  <- specification (fields + geometries + views + controls + actions + layout)
    |
Session (optional)        <- live or replay backend that emits typed updates
    |
Frontend                  <- renderer that consumes a Scene and applies SessionUpdates
```

## The Primitives

### Field

Dense labeled numpy array with named axes and coordinate metadata. The primary data primitive; see [Field Model](../concepts/field-model.md) for full semantics.

### Geometry

Structural embedding that tells renderers where data lives in space. Two kinds:

- `MorphologyGeometry` - one entry per segment: position `(n, 3)`, orientation `(n, 3, 3)`, radius, length, xloc, entity_id, section_name
- `GridGeometry` - regular 2-D grid: two named dims with 1-D coordinate arrays

Geometry holds structural and positional facts. It does not hold time-varying data; that belongs in a `Field`.

### Scene

Mutable container for a complete visualization specification:

```python
Scene(
    fields={"voltage": voltage_field},
    geometries={"morph": morphology_geom},
    views={"main": MorphologyViewSpec(...)},
    controls={"speed": ControlSpec(...)},
    actions={"reset": ActionSpec(...)},
    layout=LayoutSpec(view_3d_ids=("main",), ...),
)
```

`LayoutSpec` controls which views and controls are shown and in what order. `view_3d_ids` names the active 3D views in layout order. `view_3d_hosts` is the more generic hosting layer: it says how those views are mounted in the frontend. If the resolved 3D view list is empty, the frontend collapses to a plot-and-controls layout. `main_3d_view_id` remains as a compatibility alias for the first 3D view during the transition.

The important architectural split is:

- `ViewSpec` says what to render
- `View3DHostSpec` says how one or more 3D views are hosted
- `LayoutSpec` orders those hosts alongside other panels

See [View and Layout Model](../concepts/view-layout-model.md) for the user-facing mental model behind that split.

`Scene` is mutable. The frontend modifies `fields` in place on `FieldReplace` / `FieldAppend` and patches `views` / `controls` on `ScenePatch`.

### Session

Optional live or replay backend. A `Session`:

1. Returns an initial `Scene` from `initialize()` or `None` if the scene is provided externally
2. Steps forward on each `advance()` call
3. Receives `SessionCommand`s via `handle(command)`
4. Emits `SessionUpdate`s that `read_updates()` drains

Use `BufferedSession` for stateful sessions. It provides an `emit()` method and handles the update queue automatically.

See [Session Protocol](session-protocol.md) for the full message types.

### Frontend

The VisPy/PyQt6 frontend consumes a `Scene` and updates panels in response to `SessionUpdate`s. It owns all UI state: selection, slice position, and control values. It sends semantic commands to the session, never raw GUI events.

See [VisPy Frontend](vispy-frontend.md) for panel structure and refresh planning, and [View and Layout Model](../concepts/view-layout-model.md) for the higher-level composition model.

## Architectural Axes

The public model needs to stay cleanly factored along three independent axes:

- backend/runtime
  - examples: NEURON, Jaxley, replay
- features
  - examples: controls, live traces, morphology, surfaces, actions
- layout
  - examples: default split view, plot-only arrangement, future workbench/docking arrangements

These axes should not be conflated.

A NEURON-backed app is not automatically a morphology app. A signaling cascade viewer can be a NEURON-backed live app with controls and traces but no morphology. Likewise, a morphology view is not conceptually tied to only one backend.

Convenience builders may exist for current workflows, but they should be understood as helpers over the common model, not as separate conceptual app families.

## Key Invariants

**Field is the data primitive.** Do not introduce `TimeSeries`, `SurfaceData`, or similar types. A `Field` plus a `ViewSpec` is always sufficient.

**Geometry is structural, Field is dynamic.** Positions and connectivity go in `Geometry`. Measured values go in `Field`. Never put time-varying data in `Geometry`.

**Frontends own UI state.** Selection, slice position, and control values live in the frontend's `state` dict. Backends receive `SetControl` / `InvokeAction`, not raw GUI events.

**Prefer the narrowest correct update.** Use `FieldAppend` for incremental live history updates along one dimension. Use `ScenePatch` for metadata, view property, or control changes that do not require rebuilding the full scene. Use `FieldReplace` when replacing a field wholesale. Full replacements are valid, but they are the explicit broader-cost path, not the default.

**Live display state and captured history are different concerns.** A heavy live scene often wants:

- a latest-state field for current morphology or surface coloring
- an optional history field for retrospective trace inspection, playback, or replay

These should not be forced into the same default storage and update path. History capture is a feature that should be enabled explicitly when needed.

The current shared policy is `HistoryCaptureMode`:

- `ON_DEMAND` by default
- `FULL` when the app explicitly needs full all-entity history

Display fields must remain generic. A morphology or surface view may need to visualize:

- voltage
- calcium or other concentrations
- gating variables
- categorical or annotation-driven colors
- NeuroML-derived metadata
- analysis outputs computed after the simulation

So the architecture should treat "display" and "history" as field roles, not as voltage-specific concepts. Default ids such as `segment_display` and `segment_history` are role-based conventions, not claims that the displayed quantity is inherently voltage.

**`ViewSpec` expresses intent, renderers implement it.** Adding a new visualization means adding a `ViewSpec` subclass and a corresponding renderer or panel, not coupling domain logic into the renderer.

**Higher-level authoring should compose by feature, not by app type.** Convenience builders may exist for common workflows, but they should assemble the same core model. A user should be able to add traces, morphology, surfaces, and controls to one app by combining declarations, not by switching to a different top-level app abstraction.

**Backend choice is orthogonal to feature choice.** Helpers such as `build_neuron_app(...)` are current convenience APIs, not the intended long-term architectural boundary. "Uses NEURON" should not imply "shows morphology," and "shows morphology" should not imply one backend-specific app type.
