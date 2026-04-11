---
name: check-concept-coverage
description: Audit whether every major conceptual area in CompNeuroVis has a concept doc in docs/concepts/. Use in the pr-readiness pipeline when a new major primitive, model, or user-facing idea is introduced.
---

# Check Concept Coverage

Run this when a new major primitive, interaction model, or user-facing idea is introduced — not just a new API name, but something a user needs a mental model for to use the library effectively.

## What is a concept doc

A concept doc (lives in `docs/concepts/`) explains a major idea a user needs to understand — not "here is the API" (that's architecture) and not "here are the steps" (that's a tutorial), but "here is the mental model." It is written for users, not contributors or agents.

## Required conceptual areas

Every area below must have a concept doc. Check `docs/concepts/` against this list:

| Conceptual area | What it should explain |
|---|---|
| Field model | What a Field is, its axes, frozen semantics, `with_values`, `select` |
| Session and update model | What a Session is, live vs replay, the update types (SceneReady, FieldReplace, FieldAppend, ScenePatch), when to use each |
| Geometry types | MorphologyGeometry vs GridGeometry, what each represents, when to use each |
| Controls, actions, and state | ControlSpec, ActionSpec, StateBinding — how user interaction maps to session commands and frontend state |
| View and layout model | What a ViewSpec is, how it differs from Field/Geometry, how LayoutSpec and View3DHostSpec compose views, and how StateBinding affects view properties |

## Steps

1. List all files in `docs/concepts/`.
2. For each required conceptual area in the table above, check whether a concept doc exists that covers it.
3. Report gaps. For each missing concept doc, state:
   - What it should explain (use the table above).
   - Which existing doc (architecture or tutorial) already contains the raw content that could be distilled into it, if any.
4. If a new primitive or model was introduced by the current change and it does not fit any existing conceptual area, determine whether it warrants a new entry in the table and a new concept doc.

## Passing threshold

All required conceptual areas must have a concept doc. Gaps are required work unless explicitly recorded as named deferred gaps with a reason in the readiness report.

## What not to do

- Do not count architecture docs as concept docs — they serve different audiences.
- Do not create a concept doc that just restates the architecture doc; distill the mental model a user needs, not internal implementation details.
- Do not add to the required table speculatively — only add an area when a real user-facing concept is introduced that has no existing home.
