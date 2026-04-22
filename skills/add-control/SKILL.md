---
name: add-control
description: Add or update a control type in CompNeuroVis. Use when introducing a new spec class (scalar or compound) that the user can interact with in the controls panel, or when changing how a control maps onto frontend state.
metadata:
  kind: authoring
  surface: frontend
  stage: implement
  trust: general
---

# Add a Control Type

Read `docs/concepts/controls-actions-state.md` first.

Reference implementations: `src/compneurovis/core/controls.py` (specs) and
`src/compneurovis/frontends/vispy/panels.py` (`XYPadWidget`, `ControlsPanel._build_control_row`).

## Steps

1. **Define the spec** in `src/compneurovis/core/controls.py`.
   - Use a frozen dataclass with `slots=True`.
   - Scalar controls (one state key): follow the `ControlSpec` pattern — `id`, `kind`, `default`, `min`, `max`, optional `state_key`.
   - Compound controls (two or more state keys): define separate `resolved_*_state_key()` methods, one per axis. See `XYControlSpec` for the two-key pattern.
   - Include `send_to_session: bool = False` so the control can optionally sync to the backend.

2. **Export from core** in `src/compneurovis/core/__init__.py`.
   - Add to the import line and to `__all__`.

3. **Export from the package root** in `src/compneurovis/__init__.py`.
   - Add to the import from `compneurovis.core` and to `__all__`.

4. **Build the widget** in `src/compneurovis/frontends/vispy/panels.py`.
   - Subclass `QtWidgets.QWidget`.
   - Implement `paintEvent` for custom drawing, and mouse event handlers if interactive.
   - Accept an `on_changed` callback; call it with the resolved value(s) on every change.
   - For scalar controls, add a branch inside `ControlsPanel._build_control_row`.
   - For compound controls, add a separate `ControlsPanel._build_<kind>_row` method and handle placement in `_rebuild_grid` (compound widgets typically span the full column width via `addWidget(..., 1, column_count)`).

5. **Add state initialization** in `src/compneurovis/frontends/vispy/frontend.py`, inside `_set_scene`.
   - Scalar: `self.state.setdefault(control.resolved_state_key(), control.default)`
   - Compound: `self.state.setdefault(control.resolved_x_state_key(), control.x_default)` etc.

6. **Add a change handler** in `frontend.py`.
   - Scalar controls: the existing `_on_control_changed` handles them automatically once `kind` is recognized in `ControlsPanel`.
   - Compound controls: add `_on_<kind>_changed(self, control, *values)` that updates all state keys and unions the refresh targets for each key via `refresh_planner.targets_for_state_change(key)`.

7. **Wire the handler** when constructing `ControlsPanel` (search for `ControlsPanel(` in `frontend.py`) and pass the new callback. Update `ControlsPanel.__init__` to accept and store it.

8. **Update type annotations** in any dict or function signature typed as `dict[str, ControlSpec]` that now needs to accept the new spec. Typically: `Scene.controls` in `scene.py` and `_resolved_controls_and_actions` in `frontend.py`.

## Documentation — mandatory

Every new control kind must be documented in these places:

- **`docs/concepts/controls-actions-state.md`** — primary mental-model doc. Add a subsection under `## Controls` explaining: what it is, when to choose it over a scalar control, the key fields, and a minimal code example. Update the Practical Decision Rule table at the bottom.
- **`src/compneurovis/core/README.md`** — add the spec name to the exports list.
- **`docs/api/public-api.md`** — add the class name to the `members:` list under the core module mkdocstrings block.
- **`AGENTS.md` Public API Map** — add the export name under the appropriate bullet if it is a primary user-facing type.

## Validation

```bash
python -m compileall src examples tests
pytest tests/test_frontend_bindings.py
python scripts/generate_indexes.py
python scripts/generate_indexes.py --check
python scripts/check_architecture_invariants.py
```

If the new control spec appears in `compneurovis.__all__`, also run:

```bash
python scripts/check_docs_coverage.py   # or via: pytest tests/test_docs_and_indexes.py
```
