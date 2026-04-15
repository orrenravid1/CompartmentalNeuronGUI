---
title: Session and Update Model
summary: Mental model for live sessions, replay sessions, typed updates, and history capture.
---

# Session and Update Model

A `Session` is the backend side of the app. It owns model execution. The frontend owns UI state and rendering.

Use this page when you want the mental model for live backends, replay
backends, and typed updates. If you want a concrete build path first, start
with [Build a replay app](../tutorials/build-a-replay-app.md),
[Build a NEURON session](../tutorials/build-a-neuron-session.md), or
[Build a Jaxley session](../tutorials/build-a-jaxley-session.md).

The important mental model is:

- the frontend does not call simulator internals directly
- the session does not manipulate Qt widgets directly
- they communicate through typed commands and typed updates

## One Backend Loop

Every session fits the same shape:

1. `initialize()` creates or returns the first `Scene`
2. `advance()` moves the backend forward by one backend tick
3. `handle(command)` reacts to semantic user input
4. `read_updates()` drains queued updates for the frontend

For most backends, subclass `BufferedSession`. It gives you `emit(update)` and handles the update queue.

## Optional Startup Scene

Some apps know their initial layout, controls, and placeholder field structure before the worker starts. In that case, the session class can provide:

```python
@classmethod
def startup_scene(cls) -> Scene | None:
    return ...
```

When present, `run_app(...)` uses that scene immediately so the frontend opens straight into the intended view instead of waiting on the first worker-side `SceneReady`.

This hook is for static startup structure only. Live data still comes from `initialize()`, `advance()`, and normal typed updates.

## Live vs Replay

There are two common session modes.

**Live sessions**

These run a model forward:

- `NeuronSession`
- `JaxleySession`

They usually:

- build a model in `initialize()`
- step it in `advance()`
- emit current display values and optional retained history

**Replay sessions**

These do not simulate. They play back already-existing frames:

- `ReplaySession`
- `build_replay_app(...)`

Use replay when you already have saved values or precomputed frames and want the frontend to treat them like a normal app.

## Typed Updates, Not Blobs

The frontend should learn exactly what changed, not infer it from one giant payload.

The main update types are:

- `SceneReady`
  - the initial scene is ready
- `FieldReplace`
  - replace a field wholesale
- `FieldAppend`
  - append new samples along one dimension
- `ScenePatch`
  - patch view, control, or metadata properties
- `StatePatch`
  - synchronize semantic frontend state keys
- `Status`
  - show progress or info
- `Error`
  - report a backend problem

This is the core cost model for the repo. A session should emit the narrowest correct update instead of resending full state by habit.

## Commands Are Semantic

The frontend sends semantic commands, not raw GUI details:

- `SetControl`
- `InvokeAction`
- `KeyPressed`
- `EntityClicked`
- `Reset`

That means user code can stay transport-agnostic. A session author thinks in terms of "control changed" or "segment clicked," not pipe messages or widget events.

## Display vs History

Latest display state and retained history are different concerns.

Examples:

- morphology coloring needs the latest value for each segment
- a trace plot needs retained history over time
- replay often wants full retained history because the data already exists

The current policy knob is `HistoryCaptureMode`:

- `ON_DEMAND`
  - keep current display state live, retain history only for entities the app actively requests
- `FULL`
  - retain all-entity history for retrospective selection or playback

This matters because "show me the scene now" and "let me inspect any trace later" have different storage and transport costs.

## Practical Decision Rule

Use this rule when writing or reviewing a session:

- if the backend is generating new data over time, start from a live session
- if the data already exists as frames, start from `ReplaySession`
- if the change extends one axis, prefer `FieldAppend`
- if only view/control metadata changed, prefer `ScenePatch`
- if the whole field truly changed, use `FieldReplace`

If a change cannot be described clearly in those terms, the model probably needs clarification before more code is added.

Next steps:

- Read [Controls, Actions, and State](controls-actions-state.md) for the semantic UI side of the same command/update loop.
- Read [Session Protocol](../architecture/session-protocol.md) if you want the architecture-level message taxonomy.
