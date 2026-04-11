---
title: View and Layout Model
summary: Mental model for ViewSpec, StateBinding, 3-D hosts, and how scenes compose multiple views over the same data.
---

# View and Layout Model

CompNeuroVis separates four things that are easy to blur together:

- data
- geometry
- views
- layout

If those are kept distinct, the library stays composable. If they are mixed together, every new example starts inventing its own UI model.

## The Core Split

Use this rule first:

- `Field` answers "what are the values?"
- `Geometry` answers "where do those values live?"
- `ViewSpec` answers "how should those values be rendered?"
- `LayoutSpec` answers "which views are visible, and how are panels arranged?"

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

A view does not own data. It references existing data by id.

Examples:

- a morphology view points at a `MorphologyGeometry` and a color-driving field
- a surface view points at a `GridGeometry` and a 2-D field
- a line plot points at a field and the dimensions or slices it should plot

That means the same field can be consumed by multiple views, and the same geometry can be reused across many views.

## Multiple Views Over the Same Data

Views are intentionally reusable.

Valid patterns include:

- one morphology geometry shown in two different 3-D views
- one 2-D field shown as both a surface and a line slice
- one live field used for current display while another field retains history

This is important because "same data, different view" is a normal scientific workflow, not a special case.

If you need another perspective on the same data, add another view. Do not duplicate the geometry or invent a second data model unless the underlying data is actually different.

## Layout vs Hosting

`LayoutSpec` controls panel composition.

It decides things such as:

- which 3-D views are active
- whether a line plot is present
- whether controls are present
- the order of panels

For 3-D panels there is one more layer:

- `View3DHostSpec` answers "how are one or more 3-D views mounted?"

That is separate from `ViewSpec` on purpose.

`ViewSpec` should describe the rendered content. `View3DHostSpec` should describe hosting strategy.

That includes host-level camera policy such as the initial turntable distance,
azimuth, and elevation. If a surface or morphology example should start closer
or from a different angle, that belongs on `View3DHostSpec`, not on the view
spec that describes the rendered data.

Today the built-in host is:

- one independent canvas per 3-D view

But the architecture is intended to allow future alternatives such as:

- multiple views in one shared canvas
- multiple cameras over one shared scene

without changing what a morphology or surface view means.

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
- add a new `ViewSpec` when the rendering intent is new
- change `LayoutSpec` when panel composition changes
- change `View3DHostSpec` when 3-D hosting strategy changes

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
