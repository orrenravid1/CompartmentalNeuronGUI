---
name: add-control
description: Add or update a control value contract or frontend control presentation in CompNeuroVis. Use when introducing a new value spec, changing how controls map onto frontend state, or adding a controls-panel widget.
metadata:
  kind: authoring
  surface: frontend
  stage: implement
  trust: general
---

# Add a Control Type

Read `docs/concepts/controls-actions-state.md` first.

Reference implementations: `src/compneurovis/core/controls.py` for public
specs and `src/compneurovis/frontends/vispy/panels.py` for widget dispatch.

## Model

CompNeuroVis has one public `ControlSpec`.

- `ControlSpec.value_spec` is the backend/session/state contract.
- `ControlSpec.presentation` is an optional frontend widget hint.
- `ControlSpec.resolved_state_key()` returns the single state key for the control.
- `ControlSpec.default_value()` returns the initial state value.
- `SetControl(control.id, value)` is the session command shape for every control.

Current public value specs are `ScalarValueSpec`, `ChoiceValueSpec`,
`BoolValueSpec`, and `XYValueSpec`. XY controls store and send one atomic
dictionary: `{"x": float, "y": float}`.

## Steps

1. **Define or update the value contract** in `src/compneurovis/core/controls.py`.
   - Use frozen dataclasses with `slots=True`.
   - Keep value specs semantic. Do not put widget-only fields on the value spec.
   - Put widget hints such as `steps`, `scale`, and `shape` on `ControlPresentationSpec`.

2. **Keep one state key per control.**
   - Initialize frontend state from `control.default_value()`.
   - Use `control.resolved_state_key()` for `StateBinding` refresh planning.
   - Do not split one control into peer control ids unless the user should operate them independently.

3. **Export public specs** from `src/compneurovis/core/__init__.py` and `src/compneurovis/__init__.py`.
   - Add import names and `__all__` entries together.
   - Remove obsolete exports instead of adding compatibility aliases when the change is intentionally breaking.

4. **Build or update widget dispatch** in `src/compneurovis/frontends/vispy/panels.py`.
   - Dispatch from `isinstance(control.value_spec, ...)`.
   - Infer the default presentation when `control.presentation is None`.
   - Raise a clear `ValueError` for unsupported `presentation.kind`.
   - Custom widgets should call `on_value_changed(control, value)` with the complete semantic value.

5. **Keep session handling generic** in `src/compneurovis/frontends/vispy/frontend.py`.
   - `_set_scene` should call `self.state.setdefault(control.resolved_state_key(), control.default_value())`.
   - `_on_control_changed` should update one state key, send one `SetControl`, and refresh targets for one state key.

6. **Update type annotations** that mention controls.
   - `Scene.controls` should stay `dict[str, ControlSpec]`.
   - Controls panel lists and helper return types should use `ControlSpec`, not a union of widget-specific specs.

## Documentation - mandatory

Every public control contract or widget-family change must be documented in:

- `docs/concepts/controls-actions-state.md`: mental model, value shape, state key behavior, and a minimal example.
- `src/compneurovis/core/README.md`: public spec names in the package summary.
- `docs/api/public-api.md`: mkdocstrings member list for public exports.
- `AGENTS.md`: Public API Map when the name is a primary user-facing type.

If the change removes or renames a public control term, add the old term to
`docs/architecture/invariants.json` and run the invariant checker.

## Validation

```bash
python -m compileall src examples tests
pytest tests/test_frontend_bindings.py
pytest tests/test_core_bindings.py
python scripts/generate_indexes.py
python scripts/generate_indexes.py --check
python scripts/check_architecture_invariants.py
```

If the spec appears in `compneurovis.__all__`, also run:

```bash
pytest tests/test_docs_and_indexes.py
```
