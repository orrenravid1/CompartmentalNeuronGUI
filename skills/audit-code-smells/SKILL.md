---
name: audit-code-smells
description: Detect architectural drift patterns in CompNeuroVis that pass compilation and tests but violate the intended layering model. Use after a change touches core, session, backends, or frontend code when design violations may have crept in.
metadata:
  kind: coverage
  surface: cross-cutting
  stage: verify
  trust: general
---

# Audit Code Smells

Read `AGENTS.md` and `docs/architecture/core-model.md` first.

Check each of the following patterns. Report findings with file and line number.

## 1. Import Layer Violations

The intended import direction is: `core` <- `session` <- `builders`/`backends` <- `frontends`.

- Does any `core` module import from `session`, `builders`, `backends`, or `frontends`?
- Does any `session` module import from `builders`, `backends`, or `frontends`?
- Does any `backends` or `builders` module import from `frontends`?

Use `python -m compileall src` first to confirm syntax is clean, then grep import lines.

## 2. `isinstance` Dispatch on ViewSpec/Geometry Types Outside Frontend

`isinstance` checks on `ViewSpec` or `Geometry` subclasses should only appear in the frontend panel dispatch (`_make_panel_for_cell` or equivalent). If found in `core` or `session`, that is a smell - the logic belongs in the frontend or should be expressed as a polymorphic method.

Search: `isinstance.*ViewSpec`, `isinstance.*Geometry` in `src/compneurovis/core/` and `src/compneurovis/session/`.

## 3. Hardcoded Field/View IDs Outside Builders and Backends

String literals like `"voltage"`, `"trace"`, `"history"`, `"morphology"` appearing in `src/compneurovis/frontends/` or `src/compneurovis/session/` are a smell. Those IDs should be owned by builders and backends, not embedded in the transport or rendering layer.

Exception: test files and comments are acceptable.

## 4. Frontend State Initialized from Scene Data in `_set_scene`

`self.state.setdefault(...)` calls in `_set_scene` that derive values from scene geometry or view content are a smell. Initial UI state (e.g. `selected_entity_id`) should arrive from the session as a `StatePatch` after `SceneReady`, not be computed by the frontend.

Search: `setdefault` in the `_set_scene` method of `frontend.py`.

## 5. Session `_ui_state` Holding Rendering/Display Keys

Keys in `_ui_state` that represent display intent (e.g. color, visibility, label text) rather than behavioral session state are a smell. The session should track only what it needs to respond to commands correctly; display state belongs in the frontend.

Review `_ui_state` assignments in `src/compneurovis/backends/` and `src/compneurovis/session/`.

## Reporting

For each smell found: report the file, line number, pattern matched, and why it violates the model. For each area checked and clean: say so explicitly.

If smells are found, assess whether they are safe to leave (e.g. a known transitional state already in the backlog) or require a fix before the change is merged.
