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

Use this page when you want the mental model for semantic UI wiring. If you
only want to make something runnable first, go back to
[Getting Started](../getting-started.md) or one of the [Tutorials](../tutorials/build-a-static-surface.md).

## Controls

A `ControlSpec` describes a persistent UI value such as:

- a slider
- a spinbox
- a checkbox
- a dropdown
- an XY pad

Controls usually answer:

- "what value shape does this control own?"
- "should this value also be sent to the session?"
- "is there a preferred frontend widget?"

Examples:

- stimulus amplitude
- rolling window size
- selected slice axis
- stimulus position

Controls are good for parameters that have a current value and can be adjusted repeatedly.

`ControlSpec.value_spec` is the semantic value contract. It tells sessions,
state bindings, and model targets what value shape to expect.
`ControlSpec.presentation` is optional and only hints which widget the frontend
should use.

Current value specs:

- `ScalarValueSpec`: one `float` or `int`
- `ChoiceValueSpec`: one value from a fixed string option set
- `BoolValueSpec`: one boolean
- `XYValueSpec`: one atomic `{"x": float, "y": float}` dictionary

```python
from compneurovis import ControlPresentationSpec, ControlSpec, ScalarValueSpec

ControlSpec(
    id="speed",
    label="Speed",
    value_spec=ScalarValueSpec(default=1.0, min=0.1, max=4.0, value_type="float"),
    presentation=ControlPresentationSpec(kind="slider", steps=78, scale="linear"),
    send_to_session=True,
)
```

If `presentation` is omitted, the frontend infers a default widget from the
value spec: float slider, int spinbox, checkbox, dropdown, or XY pad.

### XY Controls

An `XYValueSpec` describes a 2D parameter control where the user drags one
handle to set two values together. Use it when two parameters are tightly
coupled and exploring their interaction matters more than tuning each
independently.

```python
from compneurovis import ControlPresentationSpec, ControlSpec, XYValueSpec

ControlSpec(
    id="stimulus_position",
    label="Stimulus position",
    value_spec=XYValueSpec(
        default={"x": 0.5, "y": 0.5},
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
        x_label="X",
        y_label="Y",
    ),
    presentation=ControlPresentationSpec(kind="xy_pad", shape="square"),
    send_to_session=True,
)
```

The control has one state key, normally the control `id`. The stored and sent
value is one dictionary: `{"x": float, "y": float}`. `SetControl(control.id,
value)` is still the only session command shape.

If the presentation shape is `"circle"`, the pad clips to a circle and the
corners are unreachable. Use this when the parameter space is naturally radial
or when corner values are meaningless.

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

That is how a control can drive a visual property without the backend needing to rebuild the whole scene.

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

- use `ControlSpec` with `ScalarValueSpec`, `ChoiceValueSpec`, or `BoolValueSpec` when the user is setting one value
- use `ControlSpec` with `XYValueSpec` when two parameters should move together in a 2D space
- use `ActionSpec` when the user is triggering an intent
- use `StateBinding` when a view should follow frontend state
- use `StatePatch` when the session needs to synchronize semantic UI state
- use `AttributeRef` and `SeriesSpec` when you want a declarative mapping onto model attributes

If a user has to think about pipes, widget objects, or frontend-only bookkeeping just to wire a slider or trace, the abstraction is still too low-level.

Next steps:

- Read [View and Layout Model](view-layout-model.md) for how state bindings reach visible views and panels.
- Read [Session and Update Model](session-update-model.md) for how sessions send `StatePatch` and receive semantic commands.
