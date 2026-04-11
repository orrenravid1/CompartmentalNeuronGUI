---
name: check-tutorial-coverage
description: Audit whether every primary user-facing API name has a tutorial or non-debug example. Use in the pr-readiness pipeline when a new builder, session type, view spec, or geometry type is added.
kind: coverage
surface: examples
stage: verify
trust: general
---

# Check Tutorial Coverage

Run this when a new builder function, session type, view spec, or geometry type is added to `__all__`.

## What counts as practical coverage

A primary API name is covered if it appears in either:
- `docs/tutorials/` (any tutorial), OR
- `examples/` **excluding** `examples/debug/`

A mention only in architecture docs (`docs/architecture/`) or concept docs (`docs/concepts/`) does **not** satisfy this — those explain what a thing is, not how to use it.

## Primary API names (require practical coverage)

- Builder functions: `build_*`
- Session classes: `*Session`
- View spec types: `*ViewSpec`
- Geometry types: `*Geometry`
- Core constructors used directly: `Field`, `Scene`, `ControlSpec`, `ActionSpec`, `StateBinding`, `LayoutSpec`, `grid_field`, `run_app`

## Steps

1. Read `src/compneurovis/__init__.py` and identify all primary API names using the categories above.
2. For each, search `docs/tutorials/` and `examples/` (excluding `examples/debug/`) for any mention.
3. Classify each as **Covered** or **Uncovered**.
4. For each uncovered name, state what it is and recommend the right remedy:
   - Has an architecture or concept mention but no tutorial/example → write a tutorial or add to an existing one.
   - Has a non-debug example but no tutorial → consider whether the example is sufficient or a tutorial would help users discover it.
   - Neither → both are needed.

## Passing threshold

All primary API names must have practical coverage before a change is doc-complete. Gaps are required work unless explicitly recorded as named deferred gaps with a reason in the readiness report.

## What not to do

- Do not count `examples/debug/` as practical coverage — debug examples are for development, not learning.
- Do not count architecture or concept doc mentions as practical coverage.
