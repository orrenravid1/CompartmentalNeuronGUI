---
title: Roadmap
summary: Active planning doc — phases, transition targets, next steps, definition of done, and benchmark apps.
---

# Roadmap

The living planning document for CompNeuroVis. Captures what phase the project is in, what architectural mismatches need resolving, the prioritized next steps, and which example apps must keep working.

For *why* decisions were made, see [Design Decisions](decisions.md). For deferred features and infrastructure ideas, see [Backlog](backlog.md).

## Current Architectural Direction

Stable governing principles. These should not change without deliberate discussion.

- Keep `Field` as the primary data primitive.
- Keep `Scene + optional Session + Frontend + Transport` as the top-level split.
- Treat the core model and session protocol as an internal runtime substrate, not as the default scientific authoring surface.
- Support multiple public app modes over that substrate: native simulator attachment, live simulation apps, static/replay visualization, document/editor workflows, and headless/export workflows.
- Keep NEURON and other simulator users in their native programming model. CompNeuroVis should attach to simulator objects, observe references, expose controls, add tools, and render views rather than replacing simulator code with a framework-owned DSL.
- Treat `NeuronSession` and other simulator sessions as integration implementations, not as the conceptual root of the whole framework.
- Keep frontend state owned by the frontend, not by sessions.
- Prefer typed append/patch messages over bundled full-state replacements when only part of the state changed.
- Prefer builder-driven and attachment-style simple entrypoints for common workflows.
- Prefer library-level cross-platform behavior over requiring unusual user script patterns.
- Keep backend choice, feature choice, and layout choice orthogonal.
- Keep declarative builders optional and composable. They should lower into the same substrate as native attachment APIs, not create isolated app families.

## Current Transition Targets

The main architectural mismatches still present in code. If work resumes after a pause, start here rather than scanning the whole backlog for clues.

1. Build a native-attachment and feature-composable public authoring layer
   Current issue:
   real apps such as signaling-cascade and the external pharynx research workflow still expose too much `Scene`/session plumbing for the intended scientific user.
   Needed direction:
   users should keep writing native simulator or scientific Python code, then attach CompNeuroVis features such as sections, traces, controls, tools, and layout declarations without needing to think about transport or low-level scene assembly.
   Near-term pressure point:
   trace declarations should create recorders, fields, views, panels, retention policy, append behavior, and reset behavior automatically. Built-in behaviors such as reset/pause should become declarative capabilities with default buttons, shortcuts, and command wiring rather than ad hoc per-example or per-session action plumbing.

2. Separate runtime substrate from app modes
   Current issue:
   `Session` currently carries too much conceptual weight: it is both the shared runtime/update mechanism and the visible shape of several different app categories.
   Needed direction:
   distinguish the shared runtime/event substrate from sibling public modes such as `StaticApp`, `LiveApp`, simulator attachment/viewer APIs, document/editor apps, and headless/export flows.
   Near-term pressure point:
   a NeuroML editor, a static replay viewer, and a live NEURON tuning app should share scene/update/frontend infrastructure without pretending to be subclasses of the same simulator-facing workflow.

3. Separate simulation cadence from presentation cadence
   Current issue:
   batching is currently doing double duty as both throughput control and perceived smoothness control.
   Needed direction:
   keep backend stepping, latest-state delivery, history capture, and playback/presentation smoothing as separable concerns.

4. Replace the transitional layout shell with a generic workbench model
   Current issue:
   splitters and fixed panel slots are useful, but still encode a temporary layout model.
   Needed direction:
   one explicit panel model plus generic panel composition, with uniform panel ids across 3-D, plots, state graphs, and controls, future saved layouts, and richer panel arrangements.
   Active proposal:
   [Layout Workbench Proposal](proposals/layout-workbench-proposal.md)

5. Formalize replay/history semantics across backends
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
- stock NEURON examples, static surfaces, cross-sections, signaling cascade, and external pharynx-style workflows have all been exercised on the new stack
- docs scaffold, `AGENTS.md`, skill catalog, and generated indexes are in place

Remaining edge work inside this phase:

- continue tightening rough edges found while using real apps
- keep performance parity and cross-platform behavior stable as other phases begin

### Phase 2: Native Attachment, App Modes, and Better Public Authoring Surface

Status: not complete

Scope:

- add native simulator attachment APIs, starting with NEURON, so users can construct simulator models normally and then register sections, traces, controls, tools, and layout features
- split the public app model into sibling modes over the same substrate: native simulator attachment, live simulation, static/replay, document/editor, and headless/export
- replace the transitional fixed layout with a genuinely generic workbench/layout model
- move closer to Blender/Unity/Unreal-style panel composition
- reduce framework exposure in the public authoring API
- formalize startup-scene behavior
- continue improving live plotting and interaction ergonomics without one-off app logic

Target outcomes:

- generic layout system with a default arrangement and customizable panel composition
- better builder/default and native-attachment APIs so domain users rarely touch scene plumbing directly
- feature-composable high-level authoring:
  - adding a trace plot, morphology view, surface view, controls, or a reusable tool should mean adding another feature declaration
  - users should not have to switch to a different internal protocol model because they want one more visualization mode
- tiered authoring surface:
  - native attachment APIs for users who already have simulator objects
  - very declarative defaults for common scientific workflows
  - light customization via small semantic hooks and callbacks
  - full escape hatches for advanced users
- cleaner interaction model based on strong defaults plus small semantic hooks
- reusable tools for common interaction modes such as trace selection and IClamp placement
- standard trace machinery that owns field/view/panel creation, recording, append updates, rolling retention, and reset replacement
- standard control binding that can target simulator object attributes, callbacks, state-store entries, or later document properties without per-example dispatch ladders
- clearer plot/view composition model once multiple plot panels and more editor-like workflows are added
- real research-facing examples such as signaling-cascade and the external pharynx workflow should read more like simple plotting or lightweight simulation code than frontend-framework code

### Phase 3: Alternate Frontends, Transports, and Editing Workflows

Status: not started

Scope:

- add alternate transports such as websocket-based transport
- support alternate frontends such as Unity
- deepen editing-oriented workflows such as NeuroML visual authoring after the Phase 2 app-mode split exists
- add more simulator/backend families beyond the current NEURON-first implementation
- add headless/export workflows for serializing scenes, rendering images, or generating static web/report artifacts without an interactive event loop

Target outcomes:

- frontend-agnostic protocol exercised by more than one frontend
- transport-agnostic session model exercised by more than one transport
- editor-style and export workflows living on the same core model rather than as separate infrastructure

## Next 5 Implementation Steps

These are the recommended next implementation steps in order. If only one thing is tackled next, start with step 1.

1. Add a NEURON native-attachment API and standard trace machinery
   Scope:
   introduce a `NeuronViewer`/`NeuronLiveApp`-style entrypoint that can attach to normal NEURON sections, refs, point processes, and callbacks. A trace declaration should produce recorder setup, `Field`, `LinePlotViewSpec`, panel defaults, `FieldAppend`, reset `FieldReplace`, and rolling-window retention internally.
   Concrete implications:
   the external pharynx-style app should add `ica` or EXP-2 state plots with one trace declaration instead of manually touching fields, views, panels, update emission, and reset handling.
   Why first:
   the architecture is now strong enough that the main remaining pain is authoring complexity, and the pharynx workflow is the clearest pressure test.

2. Extract standard control bindings and reusable tools
   Scope:
   controls should bind to target attributes, callbacks, state paths, or later document properties. Interaction modes such as IClamp placement, trace selection, brush selection, and property inspection should become reusable `Tool` objects instead of copied session state machines.
   Concrete implications:
   pharynx-style parameter sliders and IClamp placement should shrink to declarations plus small domain callbacks.
   Why second:
   traces solve the most visible boilerplate first; controls and tools remove the next largest source of per-app framework code.

3. Generalize the runtime into sibling app modes
   Scope:
   separate the shared command/update/state/transport runtime from public app modes: static/replay, live simulation, simulator attachment, document/editor, hybrid, and headless/export.
   Concrete implications:
   NEURON live viewing no longer acts as the implicit root ontology for future NeuroML editing or static replay.
   Why third:
   the native-attachment API will expose the required seams for splitting public app modes without prematurely abstracting around hypothetical integrations.

4. Split simulation batching from presentation smoothing
   Scope:
   separate backend stepping cadence, latest-state delivery cadence, history capture cadence, and optional playback smoothing.
   Concrete implications:
   heavy models should be able to stay efficient without forcing visibly jerky temporal playback.
   Why fourth:
   this is important, but it sits more cleanly on top of the generic display/history split and trace machinery than before it.

5. Expand display-role customization beyond the default sampled quantity
   Scope:
   make it straightforward for sessions and builders to choose other displayed quantities and color presets without redoing the backend/scene contract.
   Concrete implications:
   morphology coloring should be able to bind to calcium, gating variables, NeuroML metadata, or analysis outputs through the same role-based display/history model.
   Why fifth:
   the field-role split is in place now, so the next step is making alternate displayed quantities first-class in authoring APIs.

## Phase 2 Entry Definition Of Done

Phase 2 has meaningfully started only when all of the following are true:

- default backend builders no longer imply that morphology coloring is inherently voltage-only
- at least one native simulator attachment path exists for NEURON users who want to keep writing normal NEURON code and add CompNeuroVis features around it
- at least one higher-level, feature-composable builder path exists for common scientific apps that do not start from simulator objects
- trace declarations own field/view/panel/update/reset wiring for common line plots
- control bindings can target callbacks or simulator object attributes without a per-example `apply_control` ladder for the common case
- at least one reusable interaction tool replaces copied per-app mode state for a common workflow such as IClamp placement or trace selection
- signaling-cascade and external pharynx-style examples can be expressed without transport-aware author code
- the app-mode split is reflected in code-level abstractions, not just prose: NEURON live viewing, static/replay, and future document/editor workflows are siblings over a shared runtime substrate
- current parity examples still work while the new public authoring layer is introduced

## Benchmark Apps

These are the benchmark apps to use when validating architectural changes. If a change improves abstractions but degrades these workflows, the change is incomplete.

- signaling-cascade
  Purpose:
  benchmark for "scientific plotting + controls + live updates" without morphology being the main story.
- external pharynx research workflow
  Purpose:
  benchmark for custom interactions, multi-trace selection, mode-like workflows, and real scientific authoring pressure.
  This is a separate research codebase, so CompNeuroVis should reference it only as a high-level benchmark, not via repo-local paths or implementation-specific docs.
  Phase 2 acceptance signal:
  adding another recorded trace should be a single trace declaration, not edits to scene fields, line-plot views, panel specs, append emission, reset replacement, and max-sample bookkeeping.
- complex NEURON morphology example
  Purpose:
  benchmark for heavy live morphology rendering, update cadence, and click-to-trace behavior under load.
  Phase 2 acceptance signal:
  users can keep normal NEURON model construction and attach CompNeuroVis visualization/control features around it.
- Jaxley multicell
  Purpose:
  benchmark that the same architectural ideas work outside the NEURON backend.
- surface cross-section / animated surface examples
  Purpose:
  benchmark for display/history separation, surface rendering invalidation, and replay-like thinking outside morphology.

## Open Questions

- Should startup scenes become part of the `Session` interface or live in builders?
- Should plot configuration remain on `LinePlotViewSpec`, or do we need a higher-level plot panel model once multiple plot panels land?
- How much renderer invalidation should be explicit vs inferred from dependencies?
- What should be the stable naming for public app modes: `NeuronViewer`, `NeuronLiveApp`, `StaticApp`, `ReplayApp`, `DocumentApp`, `HeadlessApp`, or a smaller set of names?
- Where should the `Tool` abstraction sit so native simulator tools, document-editor tools, and frontend interaction tools share enough structure without forcing one ontology?
- Should document-edit update messages be introduced before the first serious NeuroML editor, or only when the editor implementation provides concrete pressure?

## Update Rule

When the project state changes:

- large multi-step feature proposal -> add a doc under `docs/architecture/design/proposals/` and link it from `backlog.md`
- phase milestone or priority shift → update this file (`roadmap.md`)
- new architectural insight or elevated lesson → add to `decisions.md` (requires deliberate review)
- new feature idea or deferred work item → add to `backlog.md` with a `Phase:` tag
