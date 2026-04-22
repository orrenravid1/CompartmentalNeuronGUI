---
title: View and Layout Model
summary: Mental model for ViewSpec, StateBinding, 3-D hosts, and how scenes compose multiple views over the same data.
---

# View and Layout Model

Use this page when you want the composition model for rendering: what a view
means, what a panel means, and how layout is kept separate from data and
geometry. If you need a runnable build path first, start with
[Build a static surface](../tutorials/build-a-static-surface.md).

CompNeuroVis separates six things that are easy to blur together:

- data
- geometry
- operators
- views
- visible panels
- layout topology

If those are kept distinct, the library stays composable. If they are mixed together, every new example starts inventing its own UI model.

## The Core Split

Use this rule first:

- `Field` answers "what are the values?"
- `Geometry` answers "where do those values live?"
- `OperatorSpec` answers "what derived operation should run over existing data?"
- `ViewSpec` answers "how should those values be rendered?"
- `PanelSpec` answers "which visible panel host exists, and what does it contain?"
- `LayoutSpec` answers "how are those panel ids arranged?"

That split is the reason one app can combine:

- morphology
- surface plots
- line plots
- controls
- actions

without needing a different app type for each combination.

## What a View Is

A `ViewSpec` is a declarative rendering intent.

Examples:

- `MorphologyViewSpec`
- `SurfaceViewSpec`
- `LinePlotViewSpec`
- `StateGraphViewSpec`

A view does not own data. It references existing data by id.

Examples:

- a morphology view points at a `MorphologyGeometry` and a color-driving field
- a surface view points at a `GridGeometry` and a 2-D field
- a line plot points at a field and the dimensions or slices it should plot
- a state graph points at named states/transitions plus node and edge fields

That means the same field can be consumed by multiple views, and the same geometry can be reused across many views.

## Multiple Views Over the Same Data

Views are intentionally reusable.

Valid patterns include:

- one morphology geometry shown in two different 3-D views
- one 2-D field shown as both a surface and a line slice
- one `GridSliceOperatorSpec` driving both a 3-D overlay and a 2-D plot
- one live field used for current display while another field retains history
- one state occupancy field and one transition field shown as a live state graph

This is important because "same data, different view" is a normal scientific workflow, not a special case.

Sometimes there is a reusable derived operation between "same data" and
"different view." That is where an operator belongs.

Example:

- `GridSliceOperatorSpec` describes how to cut a normalized slice through a 2-D field
- a 3-D host may project that slice as an overlay on top of a surface view
- a `LinePlotViewSpec` may render the operator output as a 1-D curve

That operator is not part of the surface view itself. It is a reusable
transformation over the same underlying field and geometry.

State graphs follow the same rule. `StateGraphViewSpec` stores the static
state names, node positions, and directed transitions. Dynamic values still
live in `Field` objects: `node_field_id` points at a field with dim
`("state",)`, and `edge_field_id` points at a field with dim `("edge",)`.
That makes the panel useful for finite-state machines, channel-state models,
and other state-transition systems without making the data model graph-specific.

If you need another perspective on the same data, add another view. Do not duplicate the geometry or invent a second data model unless the underlying data is actually different.

## Layout vs Hosting

`PanelSpec` is the current visible-panel seam. It declares one visible host
region, such as a 3-D panel, a line-plot panel, or a controls panel, and names
the hosted view, control, action, and overlay ids for that region.

`LayoutSpec` controls panel composition and topology over those `PanelSpec`
entries.

It decides things such as:

- which 3-D views are active
- which line plots are active
- which state graphs are active
- whether controls are present
- the order of panels

Today `LayoutSpec` already uses explicit `PanelSpec` entries for all visible
panel kinds, including 3-D, line plots, state graphs, and controls. That means
visible panel identity is already explicit and uniform at the panel-spec level.

Within that panel model, 3-D panels currently carry extra host-level settings
such as camera distance, azimuth, elevation, host kind, and projected operator
ids. That is because 3-D hosting currently has more policy than the other built
in panel kinds, not because only 3-D panels have real identities.

Today the built-in 3-D host behavior is:

- one independent canvas per hosted 3-D view

But the architecture is intended to allow future alternatives such as:

- multiple views in one shared canvas
- multiple cameras over one shared scene

without changing what a morphology or surface view means.

The current VisPy frontend also wraps line plots and the controls region in host
widgets for consistent framing. At that widget seam, the host layer and the
inner rendering/control layer are named separately on purpose so callers cannot
accidentally blur "visible panel chrome" with "inner plotting/control logic."

## StateBinding

Some view properties should follow current UI state instead of staying fixed.

`StateBinding(key)` is the mechanism for that.

Examples:

- a surface background color driven by a control
- a slice position driven by a slider
- a selected trace entity id driven by click state

This matters because a view should stay declarative. A user should be able to say "this property follows state key `x`" instead of manually pushing widget values into renderer objects.

## Morphology Color Scaling

Morphology color scaling should be configured at the view layer, not by changing the backend data contract.

The important knobs are:

- `color_field_id`
  - which field provides the values to color by
- `color_limits`
  - optional fixed `(min, max)` limits
- `color_norm`
  - `"auto"` for dynamic per-frame min/max normalization
  - `"symmetric"` for dynamic normalization symmetric around zero

That means:

- dynamic coloring remains available without changing backend updates
- fixed color limits do not require extra transport data
- a backend can keep streaming one display field while different views choose different normalization behavior

## Practical Decision Rule

When adding or reviewing a visualization change:

- add a new `Field` when the data itself is new
- add a new `Geometry` only when the spatial embedding is truly different
- add a new `OperatorSpec` when the derived operation is new but the underlying data is not
- add a new `ViewSpec` when the rendering intent is new
- change `LayoutSpec` or `PanelSpec` when panel composition changes
- change 3-D panel host properties when 3-D hosting strategy changes

If you are changing how something is shown, that is usually a view or layout problem, not a data-model problem.

If a user has to think "is this a new kind of data, or just another way of looking at the same data?" the safest default is:

- same data
- another view

## Why This Matters

This model is what keeps the library feature-composable.

It allows:

- NEURON apps without morphology
- morphology apps without line plots
- one app with both morphology and surface views
- two views of the same morphology
- future alternate frontends and host strategies

without changing the underlying data model each time.

Next steps:

- Read [Field Model](field-model.md) and [Geometry Types](geometry-types.md) for the underlying data/structure split.
- Read [Core Model](../architecture/core-model.md) for the architecture-level overview that places views, panels, layout, sessions, and frontend in one stack.
