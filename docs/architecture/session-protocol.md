---
title: Session Protocol
summary: Session lifecycle, typed commands and updates, BufferedSession pattern, and PipeTransport behavior.
---

# Session Protocol

A `Session` is the backend execution interface. The frontend drives it via commands; the backend emits typed updates.

## Lifecycle

```python
initialize()          -> Scene | None   # called once at startup
while running:
    advance()                             # one backend update tick
    read_updates() -> list[SessionUpdate]
    handle(command) on each received command
shutdown()                                # called on close
```

`is_live()` returns `True` by default. Return `False` for replay sessions that should stop after the last frame. `idle_sleep()` controls the polling interval in seconds (default `0.05`).

## BufferedSession

For most backends, subclass `BufferedSession` instead of `Session` directly. It provides:

```python
self.emit(update)        # queue a SessionUpdate
self.read_updates()      # drain the queue (handled automatically)
```

This means your backend only needs to implement `initialize()`, `advance()`, and `handle()`.

## Commands (frontend -> backend)

| Command | Fields | When sent |
|---|---|---|
| `Reset` | - | Space bar pressed |
| `SetControl` | `control_id`, `value` | Control changed with `send_to_session=True` |
| `InvokeAction` | `action_id`, `payload: dict` | Button clicked or shortcut triggered |
| `KeyPressed` | `key` | Unhandled key pressed in the frontend |
| `EntityClicked` | `entity_id` | Morphology entity clicked |
| `StopSession` | - | Window closed |

`handle(command)` receives these. Dispatch by type:

```python
def handle(self, command):
    if isinstance(command, SetControl):
        self.apply_control(command.control_id, command.value)
    elif isinstance(command, InvokeAction):
        self.apply_action(command.action_id, command.payload)
    elif isinstance(command, EntityClicked):
        self.on_entity_clicked(command.entity_id, ctx)
    elif isinstance(command, KeyPressed):
        self.on_key_press(command.key, ctx)
    elif isinstance(command, Reset):
        self.reset()
```

## Updates (backend -> frontend)

| Update | Fields | When to emit |
|---|---|---|
| `SceneReady` | `document: Scene` | Once, from `initialize()` or early in `advance()` |
| `FieldReplace` | `field_id`, `values`, `coords?`, `attrs_update?` | Replace a field wholesale |
| `FieldAppend` | `field_id`, `append_dim`, `values`, `coord_values`, `max_length?`, `attrs_update?` | Append new samples along one dimension |
| `ScenePatch` | `view_updates`, `control_updates`, `metadata_updates` | When view properties or control definitions change |
| `StatePatch` | `updates: dict[str, Any]` | Synchronize frontend state keys used by `StateBinding` |
| `Status` | `message: str`, `timeout_ms?` | Progress or info messages shown in the status bar |
| `Error` | `message: str` | Non-fatal errors shown in the status bar |

## Update Granularity Rule

The protocol should default to the narrowest typed update that correctly describes the change.

- Use append-style updates when data is extending along an axis.
- Use patch-style updates when metadata or view/control properties changed.
- Use wholesale replacement only when the change is genuinely full-field/full-object or when a narrower update would be misleading.

This is not just an optimization. It is the intended cost model for high-throughput rendering:

- backends should not assume the frontend wants full-state resends
- frontends should not assume one update implies a whole-window redraw
- transports should carry only the state the receiver needs to know

For live simulations, there is an additional semantic split to preserve:

- latest-state updates for what is being displayed right now
- optional history-capture updates for retrospective trace inspection, playback, or replay

These are related, but they are not the same requirement. A backend should be able to stream the current morphology or surface state without automatically committing to full trace-history capture for every entity.

The current shared policy knob is `HistoryCaptureMode`:

- `HistoryCaptureMode.ON_DEMAND`: keep latest display state live, retain trace history only for entities the app actively requests
- `HistoryCaptureMode.FULL`: retain full all-entity history for retrospective trace selection or playback

### FieldReplace vs FieldAppend vs ScenePatch

Use `FieldReplace` when the entire field should be replaced:

```python
self.emit(FieldReplace(field_id="segment_display", values=new_display_values))
```

Pass `coords=None` (the default) when grid coordinates are unchanged — the frontend will skip re-uploading x/y vertex data to the GPU and skip rebuilding axes. Pass explicit `coords` only when the coordinate arrays themselves change:

```python
# Value-only update — fast path, no coord re-upload, no axes rebuild
self.emit(FieldReplace(field_id="height", values=new_z))

# Coordinate update — full refresh including axes
self.emit(FieldReplace(field_id="height", values=new_z, coords={"x": new_x, "y": new_y}))
```

Use `FieldAppend` when live data should extend an existing field along one axis:

```python
self.emit(
    FieldAppend(
        field_id="segment_history",
        append_dim="time",
        values=new_segment_samples,
        coord_values=new_times,
        max_length=1000,
    )
)
```

This is the preferred path for live trace-style data because it avoids resending full history on every update.

For heavy live backends, the recommended default is:

- `FieldReplace` or equivalent latest-only updates for current display state
- `FieldAppend` only for history that the app has explicitly chosen to retain

This preserves performance while keeping retrospective history available as an opt-in feature rather than a default cost.

Use `ScenePatch` when structure or metadata changes, for example renaming a view title, updating a control range, or changing a display property without rebuilding the whole document:

```python
self.emit(ScenePatch(view_updates={"main": {"title": "updated title"}}))
```

Do not use `ScenePatch` for value updates. Do not rebuild and re-emit `SceneReady` just to change a view title.

Use `StatePatch` when the session needs to synchronize semantic UI state such as selected traces or other state keys consumed by `StateBinding`:

```python
self.emit(StatePatch({"selected_trace_entity_ids": selected_ids}))
```

This is the normal path for session-driven interaction state. User code should not need to care whether the session is running in another process.

If you find yourself repeatedly emitting large `FieldReplace`s for high-frequency changes, that is a signal to consider a narrower protocol shape rather than normal usage.

## PipeTransport

`PipeTransport` runs the `Session` in a worker process (or thread on permission error) and bridges it to the Qt event loop via a 60 Hz timer.

The transport accepts a lazy session source:

- a `Session` subclass
- a top-level zero-argument factory returning a `Session`

For worker-backed apps, this is the required shape so session construction happens inside the worker.

- Startup: spawns the worker, calls `session.initialize()`, emits `SceneReady` if it returns a `Scene`
- Poll loop: calls `session.advance()`, drains `read_updates()`, forwards updates to the frontend
- Commands: `transport.send_command(cmd)` queues a command for the worker

On Windows, `configure_multiprocessing()` must be called before spawning to handle the `spawn` start method. `run_app()` does this automatically.

## Interaction Model

For worker-backed apps, custom interaction hooks belong on the session and are driven by semantic commands:

- `InvokeAction`
- `KeyPressed`
- `EntityClicked`

This keeps transport constraints out of user code. A session can respond with:

- `StatePatch` for state-bound UI changes
- `Status` for user feedback
- normal field updates for data changes

The frontend may still support explicit frontend-side interaction targets as an advanced escape hatch, but that is not the default authoring model.

## Live Update Cadence

For live sessions, backend stepping and frontend emission do not need to be identical. A backend may advance several internal simulation steps inside one `advance()` call and then emit a single `FieldAppend` containing multiple new samples. That is the preferred pattern for high-frequency simulations where per-step IPC would be too expensive.
