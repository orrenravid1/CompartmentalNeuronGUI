---
title: Controls, Actions, and State
summary: Mental model for how UI controls, semantic actions, state bindings, and declarative attribute refs fit together.
---

# Controls, Actions, and State

CompNeuroVis separates three things that are easy to mix together:

- controls
- actions
- state

That separation is what keeps the frontend flexible without leaking transport details into user code.

## Controls

A `ControlSpec` describes a persistent UI value such as:

- a slider
- a checkbox
- a dropdown

Controls usually answer:

- "what value should the UI hold right now?"
- "should this value also be sent to the session?"

Examples:

- stimulus amplitude
- rolling window size
- selected slice axis

Controls are good for parameters that have a current value and can be adjusted repeatedly.

## Actions

An `ActionSpec` describes a semantic intent:

- reset
- arm selection mode
- add selected trace
- clear traces

Actions do not represent ongoing state by themselves. They represent something the user wants to trigger.

If you find yourself storing a long-lived boolean inside an action, that is usually a sign the value belongs in state instead.

## State

Frontend state is the current UI state dictionary.

Typical keys include:

- `selected_entity_id`
- `selected_trace_entity_ids`
- slice position keys
- control values

`StateBinding(key)` lets a view property read from that dictionary at render time.

That is how a control can drive a visual property without the backend needing to rebuild the whole document.

## Session-Driven State

The session can also update frontend state through `StatePatch`.

That is the normal pattern when the backend needs to drive semantic UI state, for example:

- changing which traces are selected
- toggling a mode after an action
- updating labels tied to the current selection

The key point is that the frontend owns the state dictionary, but the session can request semantic changes to it.

## Declarative Binding Helpers

Two helpers matter for more declarative app configuration.

### AttributeRef

`AttributeRef(owner, attribute)` is a small path object for reading or writing nested model attributes.

Example:

```python
from compneurovis import AttributeRef
AttributeRef("receptor", "kd1")
```

This means:

- read `root.receptor.kd1`
- write `root.receptor.kd1 = value`

It is useful when you want controls or bindings to target model parameters declaratively instead of hand-writing custom setter code for every case.

### SeriesSpec

`SeriesSpec` describes one named plotted series:

- `key`
- `label`
- `source`
- `color`

The `source` is an `AttributeRef`, so a series can declaratively point at a model value such as `ligand.C` or `receptor.bound1`.

That makes multi-trace scientific apps read more like plotting configuration and less like frontend plumbing.

## Practical Decision Rule

Use this rule when authoring app behavior:

- use `ControlSpec` when the user is setting a value
- use `ActionSpec` when the user is triggering an intent
- use `StateBinding` when a view should follow frontend state
- use `StatePatch` when the session needs to synchronize semantic UI state
- use `AttributeRef` and `SeriesSpec` when you want a declarative mapping onto model attributes

If a user has to think about pipes, widget objects, or frontend-only bookkeeping just to wire a slider or trace, the abstraction is still too low-level.
