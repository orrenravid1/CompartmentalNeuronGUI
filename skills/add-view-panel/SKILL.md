---
name: add-view-panel
description: Add or update a frontend view or panel in CompNeuroVis. Use when extending the VisPy frontend with a new panel, renderer, or view-spec consumer, or when changing how frontend-owned state maps onto rendered output.
---

# Add a View Panel

Read `docs/architecture/vispy-frontend.md` first.

Reference implementations: `src/compneurovis/frontends/vispy/panels.py` and `src/compneurovis/frontends/vispy/renderers.py`.

1. Define a `ViewSpec` subclass in `src/compneurovis/core/views.py` that captures rendering intent.
2. Implement the panel or renderer under `src/compneurovis/frontends/vispy/`; consume `Scene` and typed updates — no raw backend callbacks.
3. Keep all UI state (selection, slice position, etc.) in the frontend; do not leak it to the backend.
4. Wire the panel into `VispyFrontendWindow` and register it with `RefreshPlanner`.
5. Update the frontend package `README.md`.
6. Add at least one test or example that exercises the new panel end-to-end.

