---
title: Layout Workbench Proposal
summary: Replace the transitional mixed layout shell with explicit panel specs, a recursive split-tree workbench model, and uniform panel identity across 3-D, plots, state graphs, and controls.
---

# Layout Workbench Proposal

Status: proposal

This document captures the active plan for replacing the transitional
`LayoutSpec.panel_grid` shell with a more generic workbench model. It is not a
settled decision yet. When the implementation stabilizes and the durable lesson
is clear, the final doctrine should move into [Design Decisions](../decisions.md).

For the current high-level priority, see [Roadmap](../roadmap.md). For the
deferred summary entry, see [Backlog](../backlog.md#frontend-layout-system).

## Problem

- `LayoutSpec.panel_grid` can only express flat rows of cells.
- That row-major shape cannot express nested split topologies such as one tall
  morphology panel beside a stacked trace-plus-controls column.
- The frontend currently builds one vertical outer `QSplitter` plus one
  horizontal splitter per row. That keeps resizing simple, but it hardcodes the
  layout topology into "rows of panels" instead of a general composition model.
- Current layout topology is still split across `panels`, `panel_grid`, and
  frontend auto-derivation instead of one fully explicit recursive layout model.
- Panel identity is already explicit through `PanelSpec`, but panel composition
  and panel topology are still only partially explicit.
- The current architecture already supports multiple 3-D views and multiple
  line plots.
- The remaining gap is that the shell around those panels is still row-major,
  so richer nested arrangements, explicit sizing policy, persistent saved
  layouts, and eventually docking/collapsing still need a stronger model.

## Goals

- Support nested horizontal and vertical splits, not only row-major grids.
- Keep one explicit panel model for every visible region: 3-D, line plot, state
  graph, and controls.
- Make row and column sizing explicit, draggable, and stable under window
  resize.
- Preserve the current architectural split:
  - `ViewSpec` describes rendered content.
  - panel specs describe visible hosts and panel-local policy.
  - layout describes panel composition.
- Remove implicit layout derivation and magic panel sentinels from the
  canonical model.
- Keep authored default layout separate from frontend-owned runtime splitter
  state.
- Leave room for future saved layouts, docking, collapsing, tabs, and shared
  3-D hosts.

## Non-Goals For The First Step

- Full Blender-style arbitrary screen graph editing.
- Dockable tabs or floating windows.
- Collapsible panels.
- Runtime layout editing commands in the session protocol.
- Changing the core `Field` / `Scene` model outside layout/panel concerns.

## What Other Platforms Do

### Blender

- Blender exposes areas as rectangular partitions that users can split, join,
  and resize directly.
- In source, screen layout is stored as vertices, edges, and rectangular areas
  (`ScrVert`, `ScrEdge`, `ScrArea`), not as a simple row/column grid.
- Area splitting is axis-based with a split factor and minimum-size checks
  before the new rectangles are committed.
- Joining keeps the layout rectangular; areas must match on the join axis so
  the merged result stays a rectangle.

Pros for CompNeuroVis:

- Very flexible topology.
- Strong direct-manipulation model for editor-like workflows.
- Naturally supports arbitrary rectangular partitions.

Cons for CompNeuroVis:

- More geometric and stateful than current needs.
- Higher implementation and maintenance cost than a recursive splitter tree.
- Best fit when corner-driven split/join interactions are a product goal, which
  is not the immediate priority.

### Unity

- Unity's editor layout is closer to a recursive container tree than to a flat
  grid.
- `WindowLayout` builds container nodes that can be `vertical`,
  `horizontal`, or `tabs`, each with `children` and optional per-child `size`
  values.
- Docked editor windows expose `minSize`, but the dock/split host owns final
  sizing behavior more than the leaf window does.

Pros for CompNeuroVis:

- Very close to the recursive split-tree shape already implied by the backlog.
- Simple to map onto nested Qt splitters.
- Keeps topology description separate from per-panel content.

Cons for CompNeuroVis:

- Docking and tab semantics add complexity we do not need yet.
- Leaf widget size hints alone are not enough; the host layer still needs a
  stronger sizing policy.

### Unreal

- Unreal's Slate UI uses explicit splitter widgets (`SSplitter`) with per-slot
  sizing rules.
- The relevant idea is not docking first; it is that each child slot can carry
  a sizing rule, size value, minimum size, resize behavior, and resize callback.
- Unreal also persists editor layouts separately via `FTabManager::FLayout`.

Pros for CompNeuroVis:

- Very clear sizing semantics.
- Good model for combining fractional sizing, minimum sizes, and later saved
  layouts.
- The slot-oriented split model maps well to the row/column sizing problem the
  current frontend actually has.

Cons for CompNeuroVis:

- Unreal's full editor framework includes more docking and tab-management
  machinery than CompNeuroVis needs in the short term.
- A direct translation would overbuild the first implementation.

## Recommendation

CompNeuroVis should adopt a Unity/Unreal-style recursive split tree, not a
Blender-style screen graph.

Blender is valuable as an upper bound on future flexibility and as a reminder
that real editor layouts are not flat grids. But the best immediate path is:

- recursive split topology
- explicit panel specs with stable ids
- explicit per-child sizing rules
- minimum sizes on every child
- frontend-owned runtime splitter state
- authored default layout kept separate from saved user layouts

This matches the current backlog direction toward explicit `PanelSpec` plus
`SplitSpec` while adding the missing sizing semantics needed for dynamic row
and column behavior.

## Proposed Model

### 1. Keep `PanelSpec`, replace implicit row-major topology

Keep `LayoutSpec.panels` as the canonical panel inventory and add
`LayoutSpec.panel_layout` as the canonical topology model.

Remove `panel_grid` from the final model. Remove panel auto-derivation from the
final model so authored scenes always specify both `panels` and `panel_layout`
explicitly.

This proposal does not treat backward compatibility as a design constraint. The
goal is to converge quickly on one explicit model, update first-party examples
and builders in one sweep, and let checks catch stale call sites.

### 2. Make the recursive layout tree place panel ids only

The workbench tree should place panel ids only. Every visible region already has
an explicit panel id through `PanelSpec`; the missing piece is a recursive
topology that uses those ids directly.

This keeps the split clean:

- `ViewSpec` still means rendered content
- `PanelSpec` means visible host, panel-local policy, and stable panel identity
- layout means how those panel ids are arranged

### 3. Make defaults explicit, not implicit

Builders may still provide convenience helpers, but the final `Scene` should
contain explicit `panels` and explicit `panel_layout`.

That means:

- no `"controls"` sentinel in the canonical layout model
- no frontend-only `_auto_panel_grid()` fallback in the canonical layout model
- no hidden panel derivation from scene contents

### 4. Add explicit sizing rules per split child

The first version only needs two sizing rules:

- `fraction`
  - default for 3-D hosts and line plots
  - consumes a proportional share of remaining space
- `auto`
  - useful for controls or other utility panels
  - starts from widget size hint or preferred size, then clamps

Every child should also carry:

- `min_size`
- optional initial `value`

`fixed_px` can be deferred until a real use case proves it is needed.

### 5. Keep runtime drag state in the frontend

Splitter positions are frontend-owned UI state. They should not become session
protocol state by default.

That means:

- authored `LayoutSpec.panel_layout` defines topology and initial sizing intent
- user drag adjustments live in frontend runtime state
- future saved layouts serialize frontend state separately instead of mutating
  the scene contract in place

This matches the repo rule that frontend UI state belongs to the frontend.

## Proposed API Shape

Illustrative direction only:

```python
@dataclass(slots=True)
class PanelSpec:
    id: str
    title: str | None = None


@dataclass(slots=True)
class View3DPanelSpec(PanelSpec):
    view_ids: tuple[str, ...]
    operator_ids: tuple[str, ...] = ()
    host_kind: Literal["independent_canvas"] = "independent_canvas"
    camera_distance: float | None = 200.0
    camera_elevation: float = 30.0
    camera_azimuth: float = 30.0


@dataclass(slots=True)
class LinePlotPanelSpec(PanelSpec):
    view_id: str


@dataclass(slots=True)
class ControlsPanelSpec(PanelSpec):
    control_ids: tuple[str, ...] = ()
    action_ids: tuple[str, ...] = ()
    controls_layout_policy: Literal["single_column", "auto_columns"] = "auto_columns"


@dataclass(slots=True)
class SplitChildSpec:
    node: "PanelNode"
    size_rule: Literal["fraction", "auto"] = "fraction"
    value: float = 1.0
    min_size: int = 120


@dataclass(slots=True)
class PanelRef:
    panel_id: str


@dataclass(slots=True)
class SplitSpec:
    axis: Literal["horizontal", "vertical"]
    children: tuple[SplitChildSpec, ...]


@dataclass(slots=True)
class LayoutSpec:
    title: str = "CompNeuroVis"
    panels: tuple[PanelSpec, ...] = ()
    panel_layout: SplitSpec | None = None
```

Possible authoring shape:

```python
panel_layout=SplitSpec(
    axis="horizontal",
    children=(
        SplitChildSpec(
            PanelRef(panel_id="morph_panel"),
            value=0.7,
            min_size=320,
        ),
        SplitChildSpec(
            SplitSpec(
                axis="vertical",
                children=(
                    SplitChildSpec(
                        PanelRef(panel_id="trace_panel"),
                        value=0.65,
                        min_size=180,
                    ),
                    SplitChildSpec(
                        PanelRef(panel_id="controls_panel"),
                        size_rule="auto",
                        min_size=120,
                    ),
                ),
            ),
            value=0.3,
            min_size=220,
        ),
    ),
)

panels=(
    View3DPanelSpec(id="morph_panel", view_ids=(MORPH_VIEW_ID,)),
    LinePlotPanelSpec(id="trace_panel", view_id=TRACE_VIEW_ID),
    ControlsPanelSpec(
        id="controls_panel",
        control_ids=("speed", "gain"),
        action_ids=("reset",),
        controls_layout_policy="auto_columns",
    ),
)
```

This is only the structural direction. Exact names can change.

## Sizing Policy

The layout model needs a clear resizing contract. Recommended policy:

- On first build, normalize `fraction` children within each splitter.
- On splitter drag, only the touched sibling group changes.
- On window resize, preserve stored fractions for `fraction` children.
- `auto` children request preferred size along the active split axis, then clamp
  to `min_size`.
- After satisfying `auto` children, distribute remaining size across `fraction`
  children.
- If total minimum sizes exceed available space, clamp every child to its
  minimum and let Qt handle the unavoidable squeeze. Do not silently drop a
  panel.

Default policy by panel kind:

- 3-D hosts: `fraction`
- line plots: `fraction`
- state graphs: `fraction`
- controls: `auto` by default, but still overridable

This keeps major scientific views visually dominant while letting controls stay
compact when appropriate.

## Migration Plan

### Phase 1: Introduce the new model and cut over the repo

- Keep `LayoutSpec.panels` and add `LayoutSpec.panel_layout`.
- Remove `panel_grid`.
- Update builders, examples, docs, and tests in one refactor sweep.
- Remove panel auto-derivation from the canonical model.

### Phase 2: Cut frontend dispatch over to panel ids

- Build visible widgets from `PanelSpec.id`.
- Refresh and lookup by panel id at the host/layout seam.
- Keep `view_id` scoped to rendered content, not visible panel identity.

### Phase 3: Replace grid-specific frontend construction

- Replace the current row-builder in `VispyFrontendWindow._rebuild_panels()`
  with a recursive splitter builder.
- Keep current Qt host widgets where useful, but hang them off explicit panel
  specs instead of ad hoc layout paths.
- Store per-split runtime weights in frontend state.

### Phase 4: Add persistence for saved layouts

- Serialize frontend splitter state separately from authored `LayoutSpec`.
- Reapply saved weights only when the topology still matches.
- Treat saved layouts as frontend configuration, not session protocol state.

### Phase 5: Revisit docking, collapsing, and tabs

- Only after the split tree and sizing policy are stable.
- Treat docking as an additive host/workbench layer, not as the first step.

## Tradeoffs And Risks

- A split tree is less flexible than Blender's screen graph, but much cheaper to
  implement and enough for the layouts currently under discussion.
- One explicit panel model is cleaner, but it requires a broad refactor across
  current assumptions in builders, docs, frontend lookup, refresh plumbing, and
  tests.
- `auto` sizing is useful for controls, but overuse would make layouts unstable.
  It should stay rare and intentional.
- Saved layouts can become brittle if leaf identities are not stable. Host and
  panel ids therefore need to be treated as part of the persistence contract.

## Open Questions

- Should controls remain one logical panel kind with one recommended default
  region, or should the workbench model encourage multiple named control
  regions from day one?
- Do we need a `fixed_px` size rule, or are `fraction + auto + min_size`
  sufficient for the first implementation?
- Should runtime layout restoration happen globally per app, per example, or per
  scene title?
- How much authored layout mutability should `ScenePatch` support once
  `panel_layout` exists?

## Proposal-Doc Workflow

This layout proposal also suggests a lightweight planning convention for the
rest of the repo:

- Keep `backlog.md` as the summary list of deferred work and proposal links.
- When a feature grows into a multi-step architecture plan, give it its own doc
  under `docs/architecture/design/proposals/`.
- Link that doc from the relevant backlog item instead of turning the backlog
  into one giant design memo.
- Move only settled, durable lessons into [Design Decisions](../decisions.md).

That keeps active design work discoverable without blurring backlog, roadmap,
and decisions into one file.

## References

Internal references:

- [View and Layout Model](../../../concepts/view-layout-model.md)
- [VisPy Frontend](../../../architecture/vispy-frontend.md)
- [Roadmap](../roadmap.md)
- [Backlog](../backlog.md)

External references:

- Blender manual, areas and resizing:
  <https://docs.blender.org/manual/en/latest/interface/window_system/areas.html>
- Blender screen data types (`ScrVert`, `ScrEdge`, `ScrArea`):
  <https://github.com/blender/blender/blob/main/source/blender/makesdna/DNA_screen_types.h>
- Blender area split implementation:
  <https://github.com/blender/blender/blob/main/source/blender/editors/screen/screen_edit.cc>
- Unity editor layout source (`WindowLayout`):
  <https://github.com/Unity-Technologies/UnityCsReference/blob/master/Editor/Mono/GUI/WindowLayout.cs>
- Unity `EditorWindow.minSize`:
  <https://docs.unity3d.com/2022.3/Documentation/ScriptReference/EditorWindow-minSize.html>
- Unreal `SSplitter`:
  <https://dev.epicgames.com/documentation/en-us/unreal-engine/API/Runtime/Slate/Widgets/Layout/SSplitter>
- Unreal `ESizeRule`:
  <https://dev.epicgames.com/documentation/en-us/unreal-engine/API/Runtime/Slate/Widgets/Layout/SSplitter/ESizeRule>
- Unreal `FTabManager::FLayout`:
  <https://dev.epicgames.com/documentation/en-us/unreal-engine/API/Runtime/Slate/Framework/Docking/FTabManager/FLayout>
