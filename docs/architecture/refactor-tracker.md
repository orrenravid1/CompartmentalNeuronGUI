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
- Keep `Scene + optional Session + Frontend + Transport` as the top-level split.
- Keep frontend state owned by the frontend, not by sessions.
- Prefer typed append/patch messages over bundled full-state replacements when only part of the state changed.
- Prefer builder-driven simple entrypoints for common workflows.
- Prefer library-level cross-platform behavior over requiring unusual user script patterns.
- Keep backend choice, feature choice, and layout choice orthogonal.

## Current Transition Targets

These are the main architectural mismatches still present in code. If work resumes after a pause, start here rather than scanning the whole tracker for clues.

1. Build a genuinely feature-composable public authoring layer
   Current issue:
   real apps such as signaling-cascade and pharynx still expose too much `Scene`/session plumbing for the intended scientific user.
   Needed direction:
   users should declare features, controls, tracked series, and small hooks without needing to think about transport or low-level document assembly.

2. Separate simulation cadence from presentation cadence
   Current issue:
   batching is currently doing double duty as both throughput control and perceived smoothness control.
   Needed direction:
   keep backend stepping, latest-state delivery, history capture, and playback/presentation smoothing as separable concerns.

3. Replace the transitional layout shell with a generic workbench model
   Current issue:
   splitters and fixed panel slots are useful, but still encode a temporary layout model.
   Needed direction:
   one default layout plus generic panel composition, with future saved layouts and richer panel arrangements.

4. Formalize replay/history semantics across backends
   Current issue:
   live history capture and replay/recorded history now share the same conceptual space, but the model is only partially explicit.
   Needed direction:
   replay should feel like the natural full-history form of the same architecture rather than a side path.

## Phased Roadmap

### Phase 1: Core Architecture and Workflow Parity

Status: largely complete

Scope:

- replace the old simulation-rooted model with `Scene + optional Session + Frontend + Transport`
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
- feature-composable high-level authoring:
  - adding a trace plot, morphology view, surface view, or controls should mean adding another feature declaration
  - users should not have to switch to a different app model because they want one more visualization mode
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

## Next 3 Implementation Steps

These are the recommended next implementation steps in order. If only one thing is tackled next, start with step 1.

1. Add a feature-based high-level authoring layer for common scientific workflows
   Scope:
   let users declare controls, tracked series, morphology, surfaces, actions, and layout features without manual document plumbing.
   Concrete implications:
   signaling-cascade and pharynx-style apps should get materially shorter and stop exposing transport-aware structure.
   Why second:
   the architecture is now strong enough that the main remaining pain is authoring complexity.

2. Split simulation batching from presentation smoothing
   Scope:
   separate backend stepping cadence, latest-state delivery cadence, history capture cadence, and optional playback smoothing.
   Concrete implications:
   heavy models should be able to stay efficient without forcing visibly jerky temporal playback.
   Why third:
   this is important, but it sits more cleanly on top of the generic display/history split than before it.

3. Expand display-role customization beyond the default sampled quantity
   Scope:
   make it straightforward for sessions and builders to choose other displayed quantities and color presets without redoing the backend/document contract.
   Concrete implications:
   morphology coloring should be able to bind to calcium, gating variables, NeuroML metadata, or analysis outputs through the same role-based display/history model.
   Why third:
   the field-role split is in place now, so the next step is making alternate displayed quantities first-class in authoring APIs.

## Phase 2 Entry Definition Of Done

Phase 2 has meaningfully started only when all of the following are true:

- default backend builders no longer imply that morphology coloring is inherently voltage-only
- at least one higher-level, feature-composable builder path exists for common scientific apps
- signaling-cascade and pharynx-style examples can be expressed without transport-aware author code
- the tracker's current transition targets are reflected in code-level abstractions, not just prose
- current parity examples still work while the new public authoring layer is introduced

## Confirmed Decisions

### Cross-platform launch behavior

- User-facing launch code should work the same on Windows, Linux, and macOS.
- `run_app(...)` must protect against spawned child imports internally.
- `if __name__ == "__main__":` is allowed in user scripts, but should not be required by the library just to make examples work.
- Session construction should be lazy by default for worker-backed apps.
- The recommended public launch pattern is to pass a session class or top-level zero-argument factory, not an already-created session instance.
- Eager session instances should not be supported for worker-backed apps.

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
- 3-D hosting should be a swappable layout concern:
  - `ViewSpec` should describe what to render
  - a separate host layer should describe whether views use independent canvases, shared canvases, or future shared-scene hosts
  - the current one-view-one-canvas behavior is a host implementation, not the permanent architectural assumption

### Startup layout behavior

- A live app should not visibly start in a fallback layout and then jump to the intended layout if the initial structure is already knowable.
- If layout and views are known before the session starts, provide a bootstrap `Scene` up front.

### Protocol granularity

- High-throughput rendering workflows should default to need-to-know updates, not bundled full-state pushes.
- `FieldAppend` and `ScenePatch` should be the normal path when they can express the change correctly.
- `FieldReplace` remains the full-replacement field path and should be treated as the broader-cost option.
- Full replacements are acceptable, but they should be treated as the explicit expensive path.
- Latest-state display and captured history should be modeled as separate concerns.
- The default live path should favor latest-state updates for current rendering.
- Full history capture for retrospective trace inspection or playback should be an explicit opt-in feature, not the default cost of showing a live morphology or surface.
- The shared policy name is `HistoryCaptureMode`:
  - `ON_DEMAND` for the default split between latest display state and requested trace history
  - `FULL` for all-entity history capture used by retrospective trace selection or playback
- Display roles must be field-generic rather than voltage-specific.
- Default backend document builders now use role-based display/history field ids instead of voltage-specific ids.
- Remaining near-term priority: make alternate displayed quantities and color presets easier to author without rebuilding the backend/document contract.
- The intended model is:
  - arbitrary display fields for current morphology/surface coloring
  - arbitrary history fields for retained traces or replay
  with voltage treated as one common preset, not the defining concept.
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
- The framework should expose semantic interaction hooks such as:
  - action/button invocation
  - key press
  - clicked morphology entity
- For worker-backed apps, these hooks should live on the session and be driven by semantic commands rather than frontend-only callback objects.
- Per-app interaction policy should stay outside core renderer/transport logic.

### Public authoring surface

- The simplified public API must stay feature-composable.
- High-level helpers should assemble the same underlying `Scene + Session + Frontend + Transport` model rather than introducing separate app families such as "trace app" vs "morphology app" vs "surface app".
- Backend-specific helpers such as `build_neuron_app(...)` are acceptable as transitional conveniences, but they should not become the conceptual boundary of the library.
- Backend choice, feature choice, and layout choice must stay orthogonal.
- A NEURON-backed signaling cascade app with controls and traces but no morphology is still a valid first-class app shape.
- The preferred user experience is:
  - declare controls
  - declare exposed series/fields
  - declare views/layout features
  - provide model lifecycle hooks
  while the library owns document assembly, history buffering, and protocol packaging.
- Adding morphology, surfaces, or extra plots to an existing app should mean adding declarations, not rewriting the app around a different abstraction.

## Benchmark Apps

These are the benchmark apps to use when validating architectural changes. If a change improves abstractions but degrades these workflows, the change is incomplete.

- signaling-cascade
  Purpose:
  benchmark for "scientific plotting + controls + live updates" without morphology being the main story.
- pharynx
  Purpose:
  benchmark for custom interactions, multi-trace selection, mode-like workflows, and real scientific authoring pressure.
- complex NEURON morphology example
  Purpose:
  benchmark for heavy live morphology rendering, update cadence, and click-to-trace behavior under load.
- Jaxley multicell
  Purpose:
  benchmark that the same architectural ideas work outside the NEURON backend.
- surface cross-section / animated surface examples
  Purpose:
  benchmark for display/history separation, surface rendering invalidation, and replay-like thinking outside morphology.

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
- On slow models, smaller simulation-time batches can still improve perceived smoothness even when wall-clock throughput is unchanged, because the displayed state advances in smaller temporal jumps.
- This means simulation batching is not just a throughput parameter; it is also a presentation parameter. The architecture should leave room for:
  - latest-state delivery cadence
  - history capture cadence
  - optional playback/presentation smoothing
- The original NEURON viewer had a useful split:
  - latest morphology values were streamed on the fly
  - trace history was recorded only when the user opted into it
- That split should return in typed form:
  - default live fields for current state
  - optional history fields or buffers for retrospective trace selection and replay
- Full history for every entity remains valuable for features such as click-later trace inspection and replay, but it should be configurable rather than imposed on every live session.
- The first implementation of that split currently uses voltage-specific names; this should be generalized soon so morphology coloring can bind cleanly to any field present on the geometry.

### Lazy session construction

- Worker-side error routing only applies to code that runs inside the worker.
- If a user script constructs a session eagerly at module scope, constructor side effects and constructor exceptions happen before worker error handling can help.
- The transport and builders should therefore prefer lazy session sources:
  - session class
  - top-level zero-argument factory
- Frontend-only interaction state should be a separate concern from backend session construction.
- The canonical architecture should not rely on instantiating a second full session in the UI process just to recover custom interactions.
- The chosen default is session-side semantic interaction handling:
  - `InvokeAction`
  - `KeyPressed`
  - `EntityClicked`
  with session-emitted `StatePatch` / `Status` responses as needed.

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

- `Scene`/`ViewSpec`/layout internals are acceptable framework building blocks, but they are too low-level as the primary authoring surface for domain users.
- If a user has to override document construction just to reorder controls, tune the default trace plot, or express a simple click-mode workflow, the public API is still too exposed.
- The default NEURON-style path should prefer small hook methods and simple overrides over forcing authors to manually assemble interaction machinery.
- Users must not have to reason about transport boundaries when deciding where interaction code belongs.
- If a user has to ask "does this method go in the frontend object or the session object or it breaks pipes?", the authoring model has failed.
- Refactored app examples should be treated as usability benchmarks:
  - if a pharynx or signaling-cascade app still reads like framework plumbing, the public API is not done
  - the target is code that feels comparable in complexity to a plotting script or a lightweight simulation harness

### Declarative composition vs specialized app types

- A narrowly simplified abstraction can still be wrong if it is not composable.
- A user who starts with "plot a few live traces and expose sliders" must still be able to add morphology or a surface later without changing conceptual frameworks.
- The right simplification is a feature-based authoring layer over the common core model, not separate specialized app categories with different mental models.
- Backend-labeled helpers can become misleading if they imply a default visualization mode.
- The long-term public model should read more like "choose a backend, then add features" than "pick an app type."

## Deferred Work

### Callable-based animated surface builder

- Writing an animated surface currently requires subclassing `BufferedSession` directly, which exposes session internals to users who just want to express "call this function each frame."
- The right Phase 2 primitive is a builder with a callable, `build_animated_surface_app(fn=compute_frame, ...)`, where the session is an internal implementation detail invisible to the author.
- `FuncSession` (a `BufferedSession` that calls `fn()` in `advance()`) is a valid internal implementation of that builder, but should not be a public primitive. It still requires users to think in terms of sessions.
- The builder should also accept `on_control` and `on_action` callbacks so parameter-driven computation remains expressible without a full session subclass.
- Controls that only drive visual properties via `StateBinding` should not need to involve the session at all. The builder should distinguish those from controls that require `send_to_session=True`.
- Current workaround: subclass `BufferedSession` directly, as in `examples/surface_plot/animated_surface_live.py` and `examples/surface_plot/animated_surface_replay.py`.

### Session bootstrap API

- Implemented: sessions can now provide `@classmethod startup_scene(cls) -> Scene | None`.
- `run_app(...)` uses that hook automatically when `AppSpec.scene` is absent.
- Intended use:
  - startup layout, controls, and placeholder fields known before worker start
  - open directly into the intended view without a loading-only phase
- This keeps the bootstrap path generic and avoids hand-building `AppSpec(scene=...)` in each app script.

### Plot configuration model

- `LinePlotViewSpec` now supports rolling windows and fixed y-ranges.
- Future work may need:
  - multiple synchronized plot panels
  - selectable traces
  - plot grouping
  - independent y-axes or stacked plots
  - frontend-side ring buffers for very large live fields if simple append-and-trim becomes a bottleneck
  - richer incremental plot update paths when reslicing whole fields becomes too expensive for dense multi-trace live plots
  - an explicit history-capture policy so plots can choose between:
    - current-state streaming plus selected-trace buffering
    - full all-entity history capture for retrospective selection and replay

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
  - simplify the public authoring layer so real apps like the pharynx workflow and signaling cascade can be expressed with mostly defaults, concise configuration, and small semantic callbacks

### Transitional APIs And Assumptions To Retire

- Voltage-specific default field ids as architectural concepts
- Voltage-specific default builder assumptions where they imply morphology coloring is inherently voltage-driven
- Backend-labeled builder mental models where the backend name implies the app shape
- Any user-facing workflow that requires understanding transport boundaries to place code correctly
- Temporary layout behavior that still encodes "main 3-D view plus one plot plus one control stack" as the conceptual model

## Open Questions

- Should bootstrap documents become part of the `Session` interface or live in builders?
- Should plot configuration remain on `LinePlotViewSpec`, or do we need a higher-level plot panel model once multiple plot panels land?
- How much renderer invalidation should be explicit vs inferred from dependencies?

## Update Rule

When a new issue or design insight appears:

1. Record the stable lesson here.
2. Mark whether it is implemented or deferred.
3. Link to the canonical architecture doc if the idea becomes foundational.
