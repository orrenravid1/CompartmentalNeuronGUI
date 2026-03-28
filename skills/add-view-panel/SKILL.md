---
name: add-view-panel
description: Add or update a frontend view or panel in CompNeuroVis. Use when extending the VisPy frontend with a new panel, renderer, or view-spec consumer, or when changing how frontend-owned state maps onto rendered output.
---

# Add a View Panel

Read `docs/architecture/vispy-frontend.md` first.

Keep the split clear:

- `ViewSpec` defines intent
- panels/renderers implement rendering
- frontend window coordinates document loading, transport polling, and state refresh

When adding a new panel:

- keep UI state in the frontend
- consume `Document` and typed updates instead of raw backend callbacks
- update the relevant package `README.md`
- add at least one example or test that exercises the new panel

