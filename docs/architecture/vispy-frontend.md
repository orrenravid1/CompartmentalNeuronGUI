---
title: VisPy Frontend
summary: Panel structure, refresh planning, state binding, interaction hooks, and how to extend the frontend.
---

# VisPy Frontend

The frontend is `VispyFrontendWindow` (a `QMainWindow`) with three panels:

| Panel | Class | Responsibility |
|---|---|---|
| 3D viewport | `Viewport3DPanel` | Renders morphology (capped cylinders) or surface mesh; handles entity picking |
| Line plot | `LinePlotPanel` | Renders 1-D slices or multi-series traces via pyqtgraph |
| Controls | `ControlsPanel` | Renders sliders, spinboxes, dropdowns; emits `SetControl` on change |

Layout is driven by `LayoutSpec`. The current implementation uses Qt splitters, so the main panes are draggable/resizable. If `main_3d_view_id` is `None`, the viewport is hidden and the right panel fills the window.

## Refresh Planning

The frontend never redraws everything on every update. `RefreshPlanner` maps incoming changes to a minimal set of `RefreshTarget`s:

```
CONTROLS        — ControlsPanel
MORPHOLOGY      — Viewport3DPanel (morphology path)
SURFACE_VISUAL  — Viewport3DPanel (surface mesh + colormap)
SURFACE_AXES    — Viewport3DPanel (axes overlay)
SURFACE_SLICE   — Viewport3DPanel (slice plane overlay)
LINE_PLOT       — LinePlotPanel
```

Three trigger sources:

- **`FieldReplace` / `FieldAppend`** → `targets_for_field_replace(field_id)`: marks whichever panels reference that field
- **State change** (control moved, entity clicked) → `targets_for_state_change(state_key)`: marks panels with a `StateBinding` on that key
- **`DocumentPatch`** → `targets_for_view_patch(view_id, changed_props)`: marks panels based on which props changed

The full refresh on `DocumentReady` marks every target that the current document's layout requires.

## State and StateBinding

The frontend holds a `state: dict[str, Any]` that maps string keys to values. Controls, selections, and slice positions all live here.

`StateBinding(key)` is a placeholder on a `ViewSpec` property that defers to `state[key]` at render time:

```python
SurfaceViewSpec(
    ...
    background_color=StateBinding("background_color"),  # resolved at refresh
    tick_count=StateBinding("tick_count"),
)
```

At refresh, `resolve_value(view_prop, state)` replaces any `StateBinding` with its current value. This is how controls drive visual properties without the backend being involved.

## Interaction Hooks

For worker-backed apps, the default interaction flow is semantic command routing to the session:

- entity click -> `EntityClicked(entity_id)`
- unhandled key press -> `KeyPressed(key)`
- button/shortcut action -> `InvokeAction(action_id, payload)`

The worker session can then emit `StatePatch`, `Status`, and normal field updates in response.

An explicit frontend interaction target is still available as an advanced escape hatch. When such a target implements these methods, the frontend calls them on the corresponding events:

```python
def on_entity_clicked(self, entity_id: str, ctx: FrontendInteractionContext) -> bool: ...
def on_key_press(self, key: str, ctx: FrontendInteractionContext) -> bool: ...
def on_action(self, action_id: str, payload: dict, ctx: FrontendInteractionContext) -> bool: ...
```

Return `True` to consume the event (prevents default handling). `FrontendInteractionContext` provides:
- `ctx.document` — current Document
- `ctx.selected_entity_id` — currently selected entity
- `ctx.entity_info(entity_id)` — section name, xloc, label
- `ctx.set_state(key, value)` — set frontend state and trigger refresh
- `ctx.show_status(message)` — status bar message
- `ctx.invoke_action(action_id, payload)` — programmatically fire an action
- `ctx.set_control(control_id, value)` — programmatically change a control

Worker-backed apps should use lazy session sources and session-side interaction hooks by default. User code should not need to split itself across frontend and backend classes just to make pipes work.

## Adding a New Panel

See the `add-view-panel` skill for the full workflow. The key steps:

1. Add a `ViewSpec` subclass in `src/compneurovis/core/views.py`
2. Add `RefreshTarget` entries in `frontend.py` and extend `RefreshPlanner` property sets
3. Implement the panel in `src/compneurovis/frontends/vispy/panels.py`
4. Add a `_refresh_<panel>()` method in `VispyFrontendWindow` and call it from `_apply_refresh_targets()`
5. Wire visibility into `_update_panel_visibility()` and `LayoutSpec` as needed

Reference: `MorphologyRenderer` and `SurfaceRenderer` in `src/compneurovis/frontends/vispy/renderers.py`.
