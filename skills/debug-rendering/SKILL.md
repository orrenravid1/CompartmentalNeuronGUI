---
name: debug-rendering
description: Debug visual rendering issues in CompNeuroVis — wrong colors, missing geometry, blank panels, or performance problems. Use when the protocol dataflow is confirmed correct but the rendered output is wrong or absent.
---

# Debug Rendering

Read `docs/architecture/vispy-frontend.md` first.

Reference: `src/compneurovis/frontends/vispy/renderers.py` and `src/compneurovis/frontends/vispy/panels.py`.

Debug in this order:

1. **Confirm the Scene is wired correctly.** Check that the `ViewSpec` `field_id` and `geometry_id` actually exist in `document.fields` and `document.geometries`. A missing key silently skips rendering.

2. **Confirm the correct `RefreshTarget` is being triggered.** Add a temporary print in `_apply_refresh_targets()` to verify which targets fire on the expected event. If no target fires, the issue is in `RefreshPlanner` — check `targets_for_field_replace()` or `targets_for_state_change()`.

3. **For morphology issues:**
   - Validate `MorphologyGeometry` shapes: `positions (n, 3)`, `orientations (n, 3, 3)`, `radii/lengths/xlocs (n,)`, `entity_ids` length `n`
   - For color issues: check `color_field_id` points to a 1-D field (one value per segment), or that `sample_dim` is set correctly for a 2-D field so `field.select({sample_dim: -1})` produces a 1-D result
   - For blank/missing segments: check `radii` — zero-radius segments are invisible

4. **For surface issues:**
   - Check `Field` shape is `(len(y_coords), len(x_coords))` — dim order matters
   - Check `GridGeometry` dims order matches the `Field` dims order
   - For color issues: print `clim` and `field.values.min()` / `.max()` — if `clim` does not contain the data range, the surface will be a single flat color

5. **For line plot issues:**
   - Check `x_dim` exists in the `Field.dims`
   - For orthogonal slice: check `orthogonal_position_state_key` is present in `state` and its value is within the field's coordinate range

6. **For `StateBinding` issues:**
   - Print `self.state` at refresh time and confirm the expected key is present with the right type
   - A missing key resolves to `None`, which may be silently ignored or cause a renderer crash

7. **For performance issues:**
   - Check `RefreshPlanner` is not returning more targets than the event warrants — a state change that hits `SURFACE_VISUAL` on every slider move will rebuild the mesh each frame
   - Check `FieldAppend` is being used instead of `FieldReplace` for incremental live history

If the renderer itself raises an exception, reproduce with `python -m compileall src` first to rule out import errors, then run the minimal static example to isolate frontend vs backend.
