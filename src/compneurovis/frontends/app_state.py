from __future__ import annotations

import copy

from compneurovis.core.app import AppSpec, LayoutSpec
from compneurovis.core.field import Field


class AppState:
    """Per-actor mutable working state — strictly ``AppState = f(AppSpec)``.

    Two members, two tiers:

    - ``spec``: a structural working copy of the blueprint. The patch stream's
      *structural* deltas (AppSpecPatch/PanelPatch/LayoutReplace/metadata) fold
      here. These are spec->spec replacements (a ``ViewSpec`` swapped for a
      ``ViewSpec``), never values in a spec — so a working copy is the correct,
      isolated home, and the orchestrator's authoritative AppSpec is untouched.
      ``spec.data.fields`` holds ``FieldSpec`` only — declarative, never live.

    - ``fields``: the live value views, derived from the blueprint's
      ``FieldSpec`` declarations via ``materialize()``. FieldAppend/FieldReplace
      fold here. This is the only home for evolving field values; no field
      value ever lives in a spec.

    The split makes immutability of the blueprint a structural consequence:
    nothing mutable points into it.
    """

    __slots__ = ("spec", "fields", "metadata", "active_layout_id")

    def __init__(self, seed: AppSpec) -> None:
        self.spec = copy.deepcopy(seed)
        self.fields: dict[str, Field] = {
            field_id: field_spec.materialize()
            for field_id, field_spec in seed.data.fields.items()
        }
        # Live metadata: seeded from the blueprint's declared metadata, then
        # folded by AppSpecPatch.metadata_updates. The blueprint's
        # AppSpec.metadata stays the declared initial — never mutated at
        # runtime (parallel to FieldSpec.initial_values).
        self.metadata: dict = dict(seed.metadata)
        # Live active-layout selection. LayoutCatalog.active is the *declared
        # default* (blueprint, parallel to ControlSpec.default_value); the
        # current selection is state. A layout-switch (e.g. NeuroML workflow
        # stages) sets this id — readers resolve via active_layout().
        self.active_layout_id: str = seed.layout_catalog.active

    def active_layout(self) -> LayoutSpec:
        return self.spec.layout_catalog.layouts[self.active_layout_id]


__all__ = ["AppState"]
