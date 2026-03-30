---
title: Refactor Tracker
summary: Living design tracker for architectural decisions, lessons learned, and deferred work during the CompNeuroVis refactor.
---

# Refactor Tracker

This document is the running plan-and-learnings tracker for the refactor. Update it when we learn something important about architecture, UX, performance, or cross-platform behavior, even if we do not implement the resulting idea yet.

Use this document for:

- design decisions that should stay stable
- lessons learned from regressions or user feedback
- deferred work that should not be forgotten
- cross-platform behavior expectations

Do not use this document as a file-by-file changelog.

## Current Architectural Direction

- Keep `Field` as the primary data primitive.
- Keep `Document + optional Session + Frontend + Transport` as the top-level split.
- Keep frontend state owned by the frontend, not by sessions.
- Prefer typed append/patch messages over bundled full-state replacements when only part of the state changed.
- Prefer builder-driven simple entrypoints for common workflows.
- Prefer library-level cross-platform behavior over requiring unusual user script patterns.

## Phased Roadmap

### Phase 1: Core Architecture and Workflow Parity

Status: largely complete

Scope:

- replace the old simulation-rooted model with `Document + optional Session + Frontend + Transport`
- make `Field` the primary data primitive
- establish a typed session protocol
- recover practical parity for the current Python + VisPy + pipes workflows
- add agent-friendly documentation, indexes, and skills

Included outcomes:

- core model and typed protocol are in place
- incremental live updates now use `FieldAppend`
- targeted frontend invalidation is in place
- stock NEURON examples, static surfaces, cross-sections, signaling cascade, and pharynx-style workflows have all been exercised on the new stack
- docs scaffold, `AGENTS.md`, skill catalog, and generated indexes are in place

Remaining edge work inside this phase:

- continue tightening rough edges found while using real apps
- keep performance parity and cross-platform behavior stable as other phases begin

### Phase 2: Generic Workbench and Better Public Authoring Surface

Status: not complete

Scope:

- replace the transitional fixed layout with a genuinely generic workbench/layout model
- move closer to Blender/Unity/Unreal-style panel composition
- reduce framework exposure in the public authoring API
- formalize bootstrap-document behavior
- continue improving live plotting and interaction ergonomics without one-off app logic

Target outcomes:

- generic layout system with a default arrangement and customizable panel composition
- better builder/default APIs so domain users rarely touch document plumbing directly
- tiered authoring surface:
  - very declarative defaults for common scientific workflows
  - light customization via small semantic hooks and callbacks
  - full escape hatches for advanced users
- cleaner interaction model based on strong defaults plus small semantic hooks
- clearer plot/view composition model once multiple plot panels and more editor-like workflows are added
- real research-facing examples such as signaling-cascade and pharynx scripts should read more like simple plotting or lightweight simulation code than frontend-framework code

### Phase 3: Alternate Frontends, Transports, and Editing Workflows

Status: not started

Scope:

- add alternate transports such as websocket-based transport
- support alternate frontends such as Unity
- support editing-oriented workflows such as NeuroML visual authoring
- add more simulator/backend families beyond the current NEURON-first implementation

Target outcomes:

- frontend-agnostic protocol exercised by more than one frontend
- transport-agnostic session model exercised by more than one transport
- editor-style workflows living on the same core model rather than as separate infrastructure

## Confirmed Decisions

### Cross-platform launch behavior

- User-facing launch code should work the same on Windows, Linux, and macOS.
- `run_app(...)` must protect against spawned child imports internally.
- `if __name__ == "__main__":` is allowed in user scripts, but should not be required by the library just to make examples work.

### Frontend invalidation model

- Whole-window refreshes are too coarse for performance-sensitive scenes.
- The frontend should invalidate only the affected targets.
- Protocol and document updates should follow the same rule: send only the information required by the affected targets, not broad bundled refreshes by default.
- The current explicit targets are:
  - controls
  - morphology
  - surface visual
  - surface axes
  - surface slice overlay
  - line plot

### Generic layout system

- Layout should be fully generic and composable, with one default arrangement rather than hard-coded app categories.
- The user-facing model should be closer to Blender, Unity, or Unreal:
  - a default layout that works immediately
  - customizable panel arrangement when needed
  - no assumption that a 3D viewport is always primary
- The current 2D-only collapse behavior is a temporary step, not the target design.
- Multi-series line plots are a first-class need within that generic layout system, not an edge case.

### Startup layout behavior

- A live app should not visibly start in a fallback layout and then jump to the intended layout if the initial structure is already knowable.
- If layout and views are known before the session starts, provide a bootstrap `Document` up front.

### Protocol granularity

- High-throughput rendering workflows should default to need-to-know updates, not bundled full-state pushes.
- `FieldAppend` and `DocumentPatch` should be the normal path when they can express the change correctly.
- `FieldReplace` remains the full-replacement field path and should be treated as the broader-cost option.
- Full replacements are acceptable, but they should be treated as the explicit expensive path.
- The cost model should be opt-in:
  - if a backend or frontend wants broader updates, it should ask for them explicitly
  - the framework should not force broad refreshes unless the change is genuinely structural

### Architectural automation

- Important vocabulary and protocol taxonomy decisions should not live only in prose.
- If the repo decides that a term is retired or canonical, that decision should be encoded in machine-readable checks.
- Breaking terminology changes should prefer immediate convergence plus automated detection of stale names over compatibility aliases that let drift persist unnoticed.

### Public interaction API

- The internal architecture may use tools/controllers/manipulators, but the default user-facing API should not require users to think in those terms.
- For the intended audience, custom interactions should be expressible with a few small callbacks and strong defaults.
- The intended audience is closer to SciPy, matplotlib, NEURON, PyTorch, and Plotly users than engine/tool authors.
- Public authoring should therefore optimize for declarative configuration plus small semantic callbacks, not explicit controller or document assembly.
- The framework should expose semantic frontend hooks such as:
  - action/button invocation
  - key press
  - clicked morphology entity
- Per-app interaction policy should stay outside core renderer/transport logic.

## Lessons Learned

### Surface cross-section performance

- Slice changes should update only:
  - the derived line plot
  - the slice overlay
- Slice changes should not rebuild:
  - the surface mesh
  - the axes
  - unrelated panels

### Renderer update granularity

- Surface rendering should distinguish:
  - geometry updates
  - color/data updates
  - axes updates
  - overlay updates
- Long-lived visuals and caches are required for good performance.

### Live session data flow

- Live simulation backends should not resend full trace history on every update.
- Incremental live data belongs in typed append-style updates, with the frontend owning the displayed rolling history.
- Backend stepping cadence and frontend emission cadence should be separable.
- For high-frequency simulations, batching several internal simulation steps into one frontend update is the preferred design.
- The frontend should drain and apply all queued transport updates in one poll tick, then refresh affected views once from the final state rather than redrawing per message.
- The same principle should extend beyond traces:
  - patch or append whenever the changed region can be described cleanly
  - reserve bundled value replacement for cases where incremental semantics would be misleading, fragile, or more complex than the full replace

### Rename drift and compatibility shims

- Compatibility aliases can hide incomplete architectural migrations by keeping tests green while docs, skills, or generated references remain semantically stale.
- For deliberate internal taxonomy changes, the preferred repo policy is:
  - remove the old term
  - encode the old term as banned in machine-readable invariants
  - regenerate derived docs
  - let tests and invariant checks surface any missed sites

### NumPy masked divide behavior

- `np.divide(..., where=...)` without `out=...` can leave masked entries undefined and produce warnings.
- Use explicit output buffers for geometry normalization paths.

### Developer experience for custom interactions

- `Document`/`ViewSpec`/layout internals are acceptable framework building blocks, but they are too low-level as the primary authoring surface for domain users.
- If a user has to override document construction just to reorder controls, tune the default trace plot, or express a simple click-mode workflow, the public API is still too exposed.
- The default NEURON-style path should prefer small hook methods and simple overrides over forcing authors to manually assemble interaction machinery.
- Refactored app examples should be treated as usability benchmarks:
  - if a pharynx or signaling-cascade app still reads like framework plumbing, the public API is not done
  - the target is code that feels comparable in complexity to a plotting script or a lightweight simulation harness

## Deferred Work

### Callable-based animated surface builder

- Writing an animated surface currently requires subclassing `BufferedSession` directly, which exposes session internals to users who just want to express "call this function each frame."
- The right Phase 2 primitive is a builder with a callable — `build_animated_surface_app(fn=compute_frame, ...)` — where the session is an internal implementation detail invisible to the author.
- `FuncSession` (a `BufferedSession` that calls `fn()` in `advance()`) is a valid *internal* implementation of that builder, but should not be a public primitive — it still requires users to think in terms of sessions.
- The builder should also accept `on_control` and `on_action` callbacks so parameter-driven computation (e.g., a speed slider that changes `fn`'s behaviour) remains expressible without a full session subclass.
- Controls that only drive visual properties (colors, axes) via `StateBinding` should not need to involve the session at all — the builder should distinguish those from controls that require `send_to_session=True`.
- Current workaround: subclass `BufferedSession` directly, as in `examples/surface_plot/animated_surface_live.py` and `animated_surface_replay.py`.

### Session bootstrap API

- The signaling cascade retrofit showed a need for a formal session-level bootstrap document hook.
- Current workaround: construct a document before `run_app(...)` when possible.
- Desired future design:
  - a small formal API for “static document known before worker start”
  - no ad hoc per-session bootstrap pattern

### Plot configuration model

- `LinePlotViewSpec` now supports rolling windows and fixed y-ranges.
- Future work may need:
  - multiple synchronized plot panels
  - selectable traces
  - plot grouping
  - independent y-axes or stacked plots
  - frontend-side ring buffers for very large live fields if simple append-and-trim becomes a bottleneck
  - richer incremental plot update paths when reslicing whole fields becomes too expensive for dense multi-trace live plots

### Frontend layout system

- Current layout is intentionally simple and should be treated as transitional.
- Future work may need:
  - dockable panels
  - collapsible panels
  - multiple 3D views
  - multiple plot panels
  - persistent saved layouts

### Remote frontend / alternate transport

- The protocol and document model should stay frontend-agnostic.
- Future work may add:
  - websocket transport
  - Unity frontend
  - richer remote command/update semantics
  - more granular patch/update messages for remote scenes where bandwidth and serialization cost make bundled updates especially expensive

### Interaction system cleanup

- The current `ActionSpec.selection_mode` path proved useful as a capability check, but it is too rigid as the default public interaction model.
- Long-term direction:
  - keep richer controller/tool internals available
  - prefer callback-driven or declarative-simple interaction hooks for common app authoring
  - avoid baking one workflow model into `ActionSpec`
- Near-term priority:
  - simplify the public NEURON-facing authoring layer so real apps like the pharynx workflow can be expressed with mostly defaults, concise configuration, and small semantic callbacks

## Open Questions

- Should bootstrap documents become part of the `Session` interface or live in builders?
- Should plot configuration remain on `LinePlotViewSpec`, or do we need a higher-level plot panel model once multiple plot panels land?
- How much renderer invalidation should be explicit vs inferred from dependencies?

## Update Rule

When a new issue or design insight appears:

1. Record the stable lesson here.
2. Mark whether it is implemented or deferred.
3. Link to the canonical architecture doc if the idea becomes foundational.
