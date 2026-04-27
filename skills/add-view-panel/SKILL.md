---
name: add-view-panel
description: Add or update a frontend view or panel in CompNeuroVis. Use when extending the VisPy frontend with a new panel, renderer, or view-spec consumer, or when changing how frontend-owned state maps onto rendered output.
metadata:
  kind: authoring
  surface: frontend
  stage: implement
  trust: general
---

# Add a View Panel

Read `docs/architecture/vispy-frontend.md` and
`docs/concepts/view-layout-model.md` first.

Reference implementations: `src/compneurovis/frontends/vispy/panels.py` and the renderer package under `src/compneurovis/frontends/vispy/renderers/`.

1. Define a `ViewSpec` subclass in `src/compneurovis/core/views.py` that captures rendering intent.
2. Implement the panel or renderer under `src/compneurovis/frontends/vispy/`; consume `Scene` and typed updates - no raw backend callbacks.
   - For new 3-D visual families, extend the `Viewport3DPanel` primary-renderer or overlay pattern. Do not model renderer selection as another panel "mode" string.
3. Keep all UI state (selection, slice position, etc.) in the frontend; do not leak it to the backend.
4. Wire the panel into `VispyFrontendWindow` and register it with `RefreshPlanner`.
5. Update public exports when the new view is part of the authoring surface:
   - `src/compneurovis/core/__init__.py`
   - `src/compneurovis/__init__.py`
   - `AGENTS.md` Public API Map
6. Update docs in the nearest canonical places:
   - `src/compneurovis/core/README.md` for the core view/model surface
   - `src/compneurovis/frontends/vispy/README.md` for the concrete panel
   - `docs/concepts/view-layout-model.md` when a new `ViewSpec` changes the user mental model
   - `docs/architecture/vispy-frontend.md` when refresh targets, panel hosts, or frontend-owned state/cadence changes
   - `docs/architecture/session-protocol.md` when live update cadence or protocol-consumption guidance changes
   - relevant roadmap, backlog, decision, or proposal docs when they list current panel kinds, explicit refresh targets, or future layout assumptions
7. Add at least one test or example that exercises the new panel end-to-end.
8. If the new view is exported from `compneurovis.__all__`, run the
   `check-docs-coverage` workflow and make sure the export appears in an
   authored doc under `docs/`; generated reference indexes do not count.
9. If the change touches authored docs, package READMEs, `AGENTS.md`, or
   generated reference indexes, follow `update-docs-and-indexes`.

Validation:

- `python scripts/generate_indexes.py`
- `python scripts/generate_indexes.py --check`
- `python scripts/check_architecture_invariants.py` when vocabulary, panel kinds, or public terminology changed
- `pytest tests/test_frontend_bindings.py`
- `pytest tests/test_docs_and_indexes.py`
- `python -m compileall src examples tests`
