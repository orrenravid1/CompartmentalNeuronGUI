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
- Keep frontend state owned by the frontend, not by sessions.
- Prefer typed append/patch messages over bundled full-state replacements when only part of the state changed.
- Prefer builder-driven simple entrypoints for common workflows.
- Prefer library-level cross-platform behavior over requiring unusual user script patterns.
- Keep backend choice, feature choice, and layout choice orthogonal.

## Current Transition Targets

The main architectural mismatches still present in code. If work resumes after a pause, start here rather than scanning the whole backlog for clues.

1. Build a genuinely feature-composable public authoring layer
   Current issue:
   real apps such as signaling-cascade and the external pharynx research workflow still expose too much `Scene`/session plumbing for the intended scientific user.
   Needed direction:
   users should declare features, controls, tracked series, and small hooks without needing to think about transport or low-level scene assembly.
   Near-term pressure point:
   built-in behaviors such as reset/pause should become declarative capabilities with default buttons, shortcuts, and command wiring rather than ad hoc per-example or per-session action plumbing.

2. Separate simulation cadence from presentation cadence
   Current issue:
   batching is currently doing double duty as both throughput control and perceived smoothness control.
   Needed direction:
   keep backend stepping, latest-state delivery, history capture, and playback/presentation smoothing as separable concerns.

3. Replace the transitional layout shell with a generic workbench model
   Current issue:
   splitters and fixed panel slots are useful, but still encode a temporary layout model.
   Needed direction:
   one explicit panel model plus generic panel composition, with uniform panel ids across 3-D, plots, state graphs, and controls, future saved layouts, and richer panel arrangements.
   Active proposal:
   [Layout Workbench Proposal](proposals/layout-workbench-proposal.md)

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
- stock NEURON examples, static surfaces, cross-sections, signaling cascade, and external pharynx-style workflows have all been exercised on the new stack
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
- formalize startup-scene behavior
- continue improving live plotting and interaction ergonomics without one-off app logic

Target outcomes:

- generic layout system with a default arrangement and customizable panel composition
- better builder/default APIs so domain users rarely touch scene plumbing directly
- feature-composable high-level authoring:
  - adding a trace plot, morphology view, surface view, or controls should mean adding another feature declaration
  - users should not have to switch to a different app model because they want one more visualization mode
- tiered authoring surface:
  - very declarative defaults for common scientific workflows
  - light customization via small semantic hooks and callbacks
  - full escape hatches for advanced users
- cleaner interaction model based on strong defaults plus small semantic hooks
- clearer plot/view composition model once multiple plot panels and more editor-like workflows are added
- real research-facing examples such as signaling-cascade and the external pharynx workflow should read more like simple plotting or lightweight simulation code than frontend-framework code

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
   let users declare controls, tracked series, morphology, surfaces, actions, and layout features without manual scene plumbing.
   Concrete implications:
   signaling-cascade and external pharynx-style apps should get materially shorter and stop exposing transport-aware structure.
   Why first:
   the architecture is now strong enough that the main remaining pain is authoring complexity.

2. Split simulation batching from presentation smoothing
   Scope:
   separate backend stepping cadence, latest-state delivery cadence, history capture cadence, and optional playback smoothing.
   Concrete implications:
   heavy models should be able to stay efficient without forcing visibly jerky temporal playback.
   Why second:
   this is important, but it sits more cleanly on top of the generic display/history split than before it.

3. Expand display-role customization beyond the default sampled quantity
   Scope:
   make it straightforward for sessions and builders to choose other displayed quantities and color presets without redoing the backend/scene contract.
   Concrete implications:
   morphology coloring should be able to bind to calcium, gating variables, NeuroML metadata, or analysis outputs through the same role-based display/history model.
   Why third:
   the field-role split is in place now, so the next step is making alternate displayed quantities first-class in authoring APIs.

## Phase 2 Entry Definition Of Done

Phase 2 has meaningfully started only when all of the following are true:

- default backend builders no longer imply that morphology coloring is inherently voltage-only
- at least one higher-level, feature-composable builder path exists for common scientific apps
- signaling-cascade and external pharynx-style examples can be expressed without transport-aware author code
- the current transition targets are reflected in code-level abstractions, not just prose
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
- complex NEURON morphology example
  Purpose:
  benchmark for heavy live morphology rendering, update cadence, and click-to-trace behavior under load.
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

## Update Rule

When the project state changes:

- large multi-step feature proposal -> add a doc under `docs/architecture/design/proposals/` and link it from `backlog.md`
- phase milestone or priority shift → update this file (`roadmap.md`)
- new architectural insight or elevated lesson → add to `decisions.md` (requires deliberate review)
- new feature idea or deferred work item → add to `backlog.md` with a `Phase:` tag
