---
title: Session Protocol
summary: Session lifecycle, typed commands and updates, BufferedSession pattern, and PipeTransport behavior.
---

# Session Protocol

A `Session` is the backend execution interface. The frontend drives it via commands; the backend emits typed updates.

## Lifecycle

```python
initialize()          -> Document | None   # called once at startup
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
| `StopSession` | - | Window closed |

`handle(command)` receives these. Dispatch by type:

```python
def handle(self, command):
    if isinstance(command, SetControl):
        self.apply_control(command.control_id, command.value)
    elif isinstance(command, InvokeAction):
        self.apply_action(command.action_id, command.payload)
    elif isinstance(command, Reset):
        self.reset()
```

## Updates (backend -> frontend)

| Update | Fields | When to emit |
|---|---|---|
| `DocumentReady` | `document: Document` | Once, from `initialize()` or early in `advance()` |
| `FieldReplace` | `field_id`, `values`, `coords?`, `attrs_update?` | Replace a field wholesale |
| `FieldAppend` | `field_id`, `append_dim`, `values`, `coord_values`, `max_length?`, `attrs_update?` | Append new samples along one dimension |
| `DocumentPatch` | `view_updates`, `control_updates`, `metadata_updates` | When view properties or control definitions change |
| `Status` | `message: str` | Progress or info messages shown in the status bar |
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

### FieldReplace vs FieldAppend vs DocumentPatch

Use `FieldReplace` when the entire field should be replaced:

```python
self.emit(FieldReplace(field_id="voltage", values=new_voltages))
```

Use `FieldAppend` when live data should extend an existing field along one axis:

```python
self.emit(
    FieldAppend(
        field_id="voltage",
        append_dim="time",
        values=new_segment_samples,
        coord_values=new_times,
        max_length=1000,
    )
)
```

This is the preferred path for live trace-style data because it avoids resending full history on every update.

Use `DocumentPatch` when structure or metadata changes, for example renaming a view title, updating a control range, or changing a display property without rebuilding the whole document:

```python
self.emit(DocumentPatch(view_updates={"main": {"title": "updated title"}}))
```

Do not use `DocumentPatch` for value updates. Do not rebuild and re-emit `DocumentReady` just to change a view title.

If you find yourself repeatedly emitting large `FieldReplace`s for high-frequency changes, that is a signal to consider a narrower protocol shape rather than normal usage.

## PipeTransport

`PipeTransport` runs the `Session` in a worker process (or thread on permission error) and bridges it to the Qt event loop via a 60 Hz timer.

- Startup: spawns the worker, calls `session.initialize()`, emits `DocumentReady` if it returns a `Document`
- Poll loop: calls `session.advance()`, drains `read_updates()`, forwards updates to the frontend
- Commands: `transport.send_command(cmd)` queues a command for the worker

On Windows, `configure_multiprocessing()` must be called before spawning to handle the `spawn` start method. `run_app()` does this automatically.

## Live Update Cadence

For live sessions, backend stepping and frontend emission do not need to be identical. A backend may advance several internal simulation steps inside one `advance()` call and then emit a single `FieldAppend` containing multiple new samples. That is the preferred pattern for high-frequency simulations where per-step IPC would be too expensive.
