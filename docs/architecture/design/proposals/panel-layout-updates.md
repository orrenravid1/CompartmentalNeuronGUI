---
title: Runtime Panel Layout Updates
summary: PanelPatch and LayoutReplace — session-driven panel content and arrangement updates without full scene rebuild.
---

# Runtime Panel Layout Updates

## Problem

`LayoutSpec.panels` is fixed at scene creation. Sessions cannot change which controls are visible, swap panel arrangements, or support model-variant switching without emitting a full `SceneReady` — which reinitializes fields, views, operators, controls, actions, and geometry. That is wasteful: if only the panel arrangement changes, the rest of the scene state should remain cold.

The concrete trigger: scientific apps with multiple model variants each have their own control set. When the user picks a variant, the controls panel should show only that variant's parameters. No mechanism for this exists today.

## Design

Two new `SessionUpdate` types.

### `PanelPatch`

Surgical update to one panel's contents. Does not affect any other panel or any scene data.

```python
@dataclass(frozen=True, slots=True)
class PanelPatch(SessionUpdate):
    panel_id: str
    control_ids: tuple[str, ...] | None = None   # None = no change
    action_ids: tuple[str, ...] | None = None    # None = no change
    view_ids: tuple[str, ...] | None = None      # None = no change
    title: str | None = None                     # None = no change
```

**Semantics:**
- `None` = leave existing value unchanged.
- `()` = explicitly clear (e.g. remove all controls from a panel).
- Tuple of ids = replace with this set.

**Frontend behavior:** look up panel by `panel_id` in `scene.layout`, apply non-`None` changes via `dataclasses.replace`, then trigger `RefreshTarget.CONTROLS` to re-render the controls widget list. The Qt widget for the panel is kept; only its contents change.

**Primary use case:** controls panel content swap when model variant changes.

### `LayoutReplace`

Replaces the full panel arrangement (panels tuple + grid). Does not touch fields, geometries, views, operators, controls, or actions.

```python
@dataclass(frozen=True, slots=True)
class LayoutReplace(SessionUpdate):
    panels: tuple[PanelSpec, ...]
    panel_grid: tuple[tuple[str, ...], ...] = ()
```

**Frontend behavior:** replace `scene.layout.panels` and `scene.layout.panel_grid`, call `_rebuild_panels()` + `_update_panel_visibility()`, then trigger a full content refresh so new panels receive their current data.

This is a widget-tree rebuild, not a scene rebuild. Fields and render state are preserved.

**Use cases:**
- Switching between completely different panel arrangements (e.g. compact vs. expanded layout).
- Adding or removing panels at runtime (e.g. revealing a second trace panel on demand).
- Multiple controls panels, each for a different logical group.

## Frontend Reconciliation Rule

The frontend maintains a `{panel_id → widget}` dict. The id is the stability key:

| Update | Panel id stable | Panel id new | Panel id gone |
|---|---|---|---|
| `PanelPatch` | update in place | ignored | ignored |
| `LayoutReplace` | update widget contents | create widget | destroy widget |

The `RefreshPlanner` is not rebuilt on either update — it depends on views, not panels.

## Scene Helper

`LayoutSpec.patch_panel(panel_id, **changes) -> bool` applies `dataclasses.replace` to one panel in the panels tuple. Returns `True` if the panel was found. Used by the frontend when processing `PanelPatch`.

## Protocol Granularity Rule (Extended)

Following the existing rule ("use the narrowest update that correctly describes the change"):

- Use `PanelPatch` when one panel's contents change (controls list, action list, title).
- Use `LayoutReplace` when panels are added, removed, or rearranged.
- Use `SceneReady` only when fields, views, operators, controls, or geometry change structurally.

Do not emit `SceneReady` just to swap a controls panel's content. `PanelPatch` is the right update.

## What This Enables

- **Model variant switching:** session emits `ScenePatch` to update control definitions for the new variant, then `PanelPatch` to update `control_ids` on the controls panel to show only relevant controls.
- **Multiple controls panels:** `LayoutSpec.panels` already supports multiple `kind="controls"` panels. `LayoutReplace` makes it possible to add or remove them at runtime.
- **Composable panel authoring:** panels are independent units identified by id. Sessions compose layouts by choosing which panels exist and what they contain.

## What This Does Not Enable

- `PanelPatch` does not support changing a panel's `kind` or `camera_distance`. For structural panel changes, use `LayoutReplace`.
- `LayoutReplace` does not change fields, views, or operators. For those, use `ScenePatch` or `FieldReplace` in the same `read_updates()` response.
- Neither update type adds new controls or actions to the scene. Controls/actions are defined in `Scene.controls` / `Scene.actions` at scene creation or via `ScenePatch.control_updates`. `PanelPatch` only changes which existing controls are _shown_ in a panel.

## Phase

Phase 2 — part of the transition to feature-composable authoring and session-driven layout.
