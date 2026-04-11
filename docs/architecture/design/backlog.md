---
title: Backlog
summary: Deferred features, infrastructure ideas, and cleanup items. Freely editable — add new items with a Phase tag.
---

# Backlog

Deferred work items. Add new ideas here with a `Phase:` tag. For active priorities and what's in flight, see [Roadmap](roadmap.md). For the reasoning behind architectural choices, see [Design Decisions](decisions.md).

**Phase tags:** `Phase: 2` · `Phase: 3` · `Phase: infrastructure` · `Phase: indefinite`

---

## Visualization Features

### Network / Graph Plotting (2D and 3D)

Phase: 2

Explored in `scratch/vispy_graph_exploration.py`. `vispy.visuals.graphs.GraphVisual` + `NetworkxCoordinates` are viable as the rendering primitive for connectivity views.

**New geometry type: `GraphGeometry`**

- Nodes: positions array `(n, 2)` or `(n, 3)`, node ids, node type labels, optional user-facing labels.
- Edges: adjacency matrix or edge list `(m, 2)`, edge weights or types as optional metadata.
- Topology is structural and does not change frame-to-frame; time-varying node activity belongs in a `Field` with dim `("node",)`.
- A `Field` with per-node values maps cleanly onto vispy `face_color` per node — activity-based coloring requires only `FieldReplace` on the color field, not a layout or adjacency rebuild.

**New view spec: `GraphViewSpec`**

- References a `GraphGeometry` and an optional color `Field`.
- `projection`: `"2d"` (PanZoomCamera) or `"3d"` (TurntableCamera / ArcballCamera).
- `layout_algorithm`: `"spring"`, `"circular"`, `"kamada_kawai"`, `"shell"` (static, via `NetworkxCoordinates`) or `"force_directed"` (animated settling via `fruchterman_reingold`).
- `directed`: bool — enables arrow decoration on edges.
- `animate_layout`: bool — enables timer-driven force-directed settling; stops automatically when the layout converges.
- Node size, node border, edge line width, and arrow style are view-level style props, not data.

**2D vs 3D distinction**

- 2D graphs: `PanZoomCamera`, node positions normalized to `[0, 1]²`, layout computed by NetworkX or vispy's force-directed.
- 3D graphs: `TurntableCamera`, z-positions either computed by a 3D layout (e.g. `nx.spring_layout` with `dim=3`) or supplied explicitly in `GraphGeometry`. The same `GraphVisual` rendering primitive applies; only the camera and coordinate dimensionality change.
- Both cases share one `GraphViewSpec` class distinguished by the `projection` field, consistent with how other views express rendering intent rather than data-model differences.

**Live coloring without layout rebuild**

- `graph_node._node.set_data(face_color=new_colors)` updates per-node colors in place without rebuilding adjacency or re-running layout.
- The correct protocol path is `FieldReplace` on the color field; the graph panel renderer maps it to `set_data(face_color=...)`.
- `FieldAppend` is not meaningful for a per-node scalar field; `FieldReplace` is the right update type here.

**Interaction**

- Node click should emit `EntityClicked(entity_id)` using the node id, consistent with morphology click semantics.
- Selection state lives in the frontend `state` dict and can drive a `StateBinding` on `face_color` or node border for highlight rendering.

**Benchmark app**

- A connectivity viewer showing a layered circuit with per-node live activity coloring (e.g. from a Jaxley or NEURON multicell run) is the natural benchmark for this feature.

---

### Plot Configuration Model

Phase: 2

`LinePlotViewSpec` now supports rolling windows and fixed y-ranges. Future work may need:

- multiple synchronized plot panels
- selectable traces
- plot grouping
- independent y-axes or stacked plots
- frontend-side ring buffers for very large live fields if simple append-and-trim becomes a bottleneck
- richer incremental plot update paths when reslicing whole fields becomes too expensive for dense multi-trace live plots
- an explicit history-capture policy so plots can choose between current-state streaming plus selected-trace buffering vs full all-entity history capture for retrospective selection and replay

---

### Frontend Layout System

Phase: 2/3

Current layout is intentionally simple and should be treated as transitional. Future work may need:

- dockable panels
- collapsible panels
- multiple 3D views
- multiple plot panels
- persistent saved layouts

---

### Default NeuronSession Layout Always Includes a Trace Plot

Phase: 2

`NeuronSceneBuilder.build_scene` always produces both a morphology view and a trace/line-plot view, with no opt-out. For apps that do not use the plot (e.g. debug examples), the unused plot panel is visible and confusing.

The default should be opt-in: morphology-only apps should not have to suppress or work around an empty plot panel they did not ask for. The right fix belongs in Phase 2 alongside the broader feature-composable authoring layer — declaring a plot should mean adding a feature, not inheriting one unconditionally from the default builder.

---

## API / Authoring

### Callable-Based Animated Surface Builder

Phase: 2

Writing an animated surface currently requires subclassing `BufferedSession` directly, which exposes session internals to users who just want to express "call this function each frame."

The right Phase 2 primitive is a builder with a callable, `build_animated_surface_app(fn=compute_frame, ...)`, where the session is an internal implementation detail invisible to the author.

- `FuncSession` (a `BufferedSession` that calls `fn()` in `advance()`) is a valid internal implementation of that builder, but should not be a public primitive. It still requires users to think in terms of sessions.
- The builder should also accept `on_control` and `on_action` callbacks so parameter-driven computation remains expressible without a full session subclass.
- Controls that only drive visual properties via `StateBinding` should not need to involve the session at all. The builder should distinguish those from controls that require `send_to_session=True`.

Current workaround: subclass `BufferedSession` directly, as in `examples/surface_plot/animated_surface_live.py` and `examples/surface_plot/animated_surface_replay.py`.

---

### Built-In Capability Registry

Phase: 2

Examples currently expose reset/pause through a mix of protocol commands, session defaults, frontend special-casing, and direct `ActionSpec` wiring. Built-in behaviors such as reset, pause/resume, and future step/capture/export actions should become declarative capabilities rather than hand-wired actions.

Intended direction:
- apps or builders opt into capabilities
- capabilities provide default label, button presence, shortcut, and command dispatch semantics
- custom `ActionSpec` remains available for truly app-specific behavior

This should remove frontend special-casing and make common simulation affordances easy to add consistently across examples and real apps.

---

### Interaction System Cleanup

Phase: 2

The current `ActionSpec.selection_mode` path proved useful as a capability check, but it is too rigid as the default public interaction model.

Long-term direction:
- keep richer controller/tool internals available
- prefer callback-driven or declarative-simple interaction hooks for common app authoring
- avoid baking one workflow model into `ActionSpec`

Near-term priority: simplify the public authoring layer so real apps like the external pharynx workflow and signaling cascade can be expressed with mostly defaults, concise configuration, and small semantic callbacks.

---

## Infrastructure

### Remote Frontend / Alternate Transport

Phase: 3

The protocol and scene model should stay frontend-agnostic. Future work may add:

- websocket transport
- Unity frontend
- richer remote command/update semantics
- more granular patch/update messages for remote scenes where bandwidth and serialization cost make bundled updates especially expensive

---

### Repo-Local MCP Server

Phase: infrastructure

The current MCP setup (`mcp.json`) gives all agents access to external tools. A natural next step is a repo-local MCP server that exposes this repo's own tooling and domain knowledge as first-class MCP tools and resources. Any agent working in this codebase — regardless of which client — would then invoke repo scripts and query repo state through the same MCP interface rather than shelling out or reading files manually.

Candidate tools: run PR-readiness check and return structured results; run architecture invariant checks; run compile check and tests; regenerate indexes or MCP configs.

Candidate resources: skills catalog, architecture invariants, public API index, current roadmap.

Candidate domain tools: parse an SWC morphology file and return segment data; inspect a serialized Field (shape, dims, coords); list and describe available examples.

Implementation path: build as a Python stdio MCP server (no extra infrastructure required); register in `mcp.json` as a local server alongside existing external ones.

Deferral reason: external MCPs cover the most acute gaps now; the local server is a higher-effort, higher-value item that makes more sense once the public authoring API (Phase 2) is stable enough that the repo's own surface is worth exposing formally.

---

### Shared Graph Memory Across Agents

Phase: infrastructure

The current auto-memory system (MEMORY.md + per-agent flat files) is per-user and per-agent. A graph-based MCP memory server committed to the repo would address both gaps:
- typed relationships between memories allow queries like "what design chain led to HistoryCaptureMode?" in a way flat markdown cannot answer
- all agents read from the same committed file, so architectural knowledge is genuinely shared

Viable implementation path: run `@modelcontextprotocol/memory` locally, pointed at a committed `memory.json` under `.compneurovis/`. JSON is git-mergeable; for append-heavy workload, auto-merge succeeds most of the time. Agents pull before starting work, push the updated `memory.json` after a session.

Deferral reason: value scales with accumulated decisions. Worth implementing once architectural decisions accumulate faster than the roadmap's manual prose can track, or once multiple agent families are actively contributing regularly.

---

### Harness-Governed Agent Infrastructure Proposals

Phase: infrastructure

If external or non-owner contributors become common, agents should use a proposal-only model for protected infrastructure surfaces. Proposals stored as structured observation artifacts under `.compneurovis/` with lifecycle: `observed` → `accepted` → `rejected` → `implemented`. Protected surfaces include `skills/**`, `AGENTS.md`, readiness/invariant/config-generation scripts, CI workflows, governance docs, and machine-readable policy artifacts under `.compneurovis/`.

Deferral reason: current solo-owner workflow is sufficient. Revisit once external contributors are common or multiple agent/client families are contributing regularly.

---

### Potential Zensical Migration

Phase: indefinite

The current docs site should stay on `MkDocs + Material + mkdocstrings` until there is a clear stable replacement path. `Zensical` is worth revisiting later because it can consume `mkdocs.yml`, which lowers migration cost if it matures. Migration should be reconsidered only once Zensical is stable enough to preserve strict local builds, predictable local authoring, and the current generated API-doc workflow without repo-specific workarounds.

---

## Cleanup / Retirement

### Transitional APIs and Assumptions to Retire

Phase: 2

- Voltage-specific default field ids as architectural concepts
- Voltage-specific default builder assumptions where they imply morphology coloring is inherently voltage-driven
- Backend-labeled builder mental models where the backend name implies the app shape
- Any user-facing workflow that requires understanding transport boundaries to place code correctly
- Temporary layout behavior that still encodes "main 3-D view plus one plot plus one control stack" as the conceptual model

---

## Completed

### Session Startup Scene API

Implemented. Sessions can now provide `@classmethod startup_scene(cls) -> Scene | None`. `run_app(...)` uses that hook automatically when `AppSpec.scene` is absent. This allows startup layout, controls, and placeholder fields to be known before worker start, opening directly into the intended view without a loading-only phase.
