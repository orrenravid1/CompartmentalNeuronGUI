---
title: VisPy Frontend
summary: Panel structure, refresh planning, state binding, interaction hooks, and how to extend the frontend.
---

# VisPy Frontend

See [View and Layout Model](../concepts/view-layout-model.md) for the higher-level mental model. This document focuses on the concrete VisPy implementation of that model.

The frontend is `VispyFrontendWindow` (a `QMainWindow`) with three panel regions:

| Panel | Class | Responsibility |
|---|---|---|
| 3D host(s) | `IndependentCanvas3DHostPanel` -> `Viewport3DPanel` | A host mounts one or more 3D views using a concrete hosting strategy; the current built-in host uses one canvas per view |
| Line plot | `LinePlotPanel` | Renders 1-D slices or multi-series traces via pyqtgraph |
| Controls | `ControlsPanel` | Renders sliders, spinboxes, dropdowns; emits `SetControl` on change |

Layout is driven by `LayoutSpec`. The current implementation uses Qt splitters, so the main panes are draggable/resizable. `LayoutSpec.view_3d_hosts` is the primary 3D layout seam. Each `View3DHostSpec` declares:

- which 3D views it owns
- which host kind should mount them
- optional host-level titling
- optional initial camera settings such as turntable distance, azimuth, and elevation

If `view_3d_hosts` is omitted, the layout derives one independent host per `view_3d_id`. If the resolved 3D host list is empty, the 3D splitter is hidden and the right panel fills the window.

Before the first `Scene` arrives, the frontend stays in an explicit loading state rather than briefly showing an empty fallback plot layout. That avoids a visible startup jump for worker-backed live apps.

## Refresh Planning

The frontend never redraws everything on every update. `RefreshPlanner` maps incoming changes to a minimal set of `RefreshTarget`s:

```
CONTROLS                     -> ControlsPanel
MORPHOLOGY(view_id)          -> one Viewport3DPanel (morphology path)
SURFACE_VISUAL(view_id)      -> one Viewport3DPanel (surface mesh + colormap)
SURFACE_AXES(view_id)        -> one Viewport3DPanel (axes overlay)
SURFACE_SLICE(view_id)       -> one Viewport3DPanel (slice plane overlay)
LINE_PLOT                    -> LinePlotPanel
```

3D refresh targets are explicitly bound to a `view_id`, so the frontend can refresh multiple morphology and surface panels independently in the same window even when the hosting layer changes.

## 3D Hosting Layer

The frontend now separates:

- view semantics
  - `MorphologyViewSpec`, `SurfaceViewSpec`
- host semantics
  - `View3DHostSpec`
- concrete VisPy widget implementation
  - currently `IndependentCanvas3DHostPanel`

That means the current behavior:

- one 3D view
- one `SceneCanvas`
- one `ViewBox`

is no longer the only architectural shape. It is the current host implementation.

This is the intended extension point for future alternatives such as:

- multiple 3D views inside one shared canvas
- multiple camera views over a shared scene host
- other host-level composition patterns

without changing `ViewSpec`, backend sessions, or the typed refresh model.

The current independent-canvas host uses a `TurntableCamera`, so host-level
camera settings are the right place to tune starting framing for surface and
morphology examples without changing the semantics of the underlying view.

Three trigger sources:

- **`FieldReplace` / `FieldAppend`** -> `targets_for_field_replace(field_id)`: marks whichever panels reference that field
- **State change** (control moved, entity clicked) -> `targets_for_state_change(state_key)`: marks panels with a `StateBinding` on that key
- **`ScenePatch`** -> `targets_for_view_patch(view_id, changed_props)`: marks panels based on which props changed

The full refresh on `SceneReady` marks every target that the current scene's layout requires.

## State and StateBinding

The frontend holds a `state: dict[str, Any]` that maps string keys to values. Controls, selections, and slice positions all live here.

`StateBinding(key)` is a placeholder on a `ViewSpec` property that defers to `state[key]` at render time:

```python
SurfaceViewSpec(
    ...,
    background_color=StateBinding("background_color"),
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

Return `True` to consume the event. `FrontendInteractionContext` provides:

- `ctx.scene` - current `Scene`
- `ctx.selected_entity_id` - currently selected entity
- `ctx.entity_info(entity_id)` - section name, xloc, label
- `ctx.set_state(key, value)` - set frontend state and trigger refresh
- `ctx.show_status(message)` - status bar message
- `ctx.invoke_action(action_id, payload)` - programmatically fire an action
- `ctx.set_control(control_id, value)` - programmatically change a control

Worker-backed apps should use lazy session sources and session-side interaction hooks by default. User code should not need to split itself across frontend and backend classes just to make pipes work.

## Adding a New Panel

See the `add-view-panel` skill for the full workflow. The key steps:

1. Add a `ViewSpec` subclass in `src/compneurovis/core/views.py`
2. Extend `RefreshPlanner` so it can target the new panel per bound `view_id` when needed
3. Implement the panel in `src/compneurovis/frontends/vispy/panels.py`
4. Add a `_refresh_<panel>()` method in `VispyFrontendWindow` and call it from `_apply_refresh_targets()`
5. Wire visibility into `_update_panel_visibility()` and `LayoutSpec` as needed

Reference: `MorphologyRenderer` and `SurfaceRenderer` in `src/compneurovis/frontends/vispy/renderers.py`.
