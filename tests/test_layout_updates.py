from __future__ import annotations

import numpy as np
import pytest

from compneurovis.core import Field, LayoutSpec, Scene
from compneurovis.core.scene import PANEL_KIND_CONTROLS, PANEL_KIND_LINE_PLOT, PANEL_KIND_VIEW_3D, PanelSpec
from compneurovis.session import LayoutReplace, PanelPatch


def _minimal_scene(panels: tuple[PanelSpec, ...] | None = None) -> Scene:
    field = Field(
        id="x",
        values=np.zeros(2, dtype=np.float32),
        dims=("t",),
        coords={"t": np.array([0.0, 1.0], dtype=np.float32)},
    )
    layout = LayoutSpec(panels=panels or ())
    return Scene(fields={"x": field}, geometries={}, views={}, layout=layout)


# ---------------------------------------------------------------------------
# LayoutSpec.patch_panel
# ---------------------------------------------------------------------------


def test_patch_panel_updates_control_ids():
    layout = LayoutSpec(
        panels=(
            PanelSpec(id="ctrl", kind=PANEL_KIND_CONTROLS, control_ids=("a", "b"), action_ids=()),
        )
    )
    found = layout.patch_panel("ctrl", control_ids=("c",))
    assert found is True
    assert layout.panel("ctrl").control_ids == ("c",)


def test_patch_panel_updates_action_ids():
    layout = LayoutSpec(
        panels=(
            PanelSpec(id="ctrl", kind=PANEL_KIND_CONTROLS, control_ids=(), action_ids=("x",)),
        )
    )
    layout.patch_panel("ctrl", action_ids=("y", "z"))
    assert layout.panel("ctrl").action_ids == ("y", "z")


def test_patch_panel_clears_control_ids_with_empty_tuple():
    layout = LayoutSpec(
        panels=(
            PanelSpec(id="ctrl", kind=PANEL_KIND_CONTROLS, control_ids=("a",), action_ids=()),
        )
    )
    layout.patch_panel("ctrl", control_ids=())
    assert layout.panel("ctrl").control_ids == ()


def test_patch_panel_updates_title():
    layout = LayoutSpec(
        panels=(PanelSpec(id="p", kind=PANEL_KIND_LINE_PLOT, view_ids=("trace",)),)
    )
    layout.patch_panel("p", title="New Title")
    assert layout.panel("p").title == "New Title"


def test_patch_panel_does_not_affect_other_panels():
    layout = LayoutSpec(
        panels=(
            PanelSpec(id="a", kind=PANEL_KIND_CONTROLS, control_ids=("x",), action_ids=()),
            PanelSpec(id="b", kind=PANEL_KIND_CONTROLS, control_ids=("y",), action_ids=()),
        )
    )
    layout.patch_panel("a", control_ids=("z",))
    assert layout.panel("a").control_ids == ("z",)
    assert layout.panel("b").control_ids == ("y",)


def test_patch_panel_returns_false_for_unknown_id():
    layout = LayoutSpec(
        panels=(PanelSpec(id="ctrl", kind=PANEL_KIND_CONTROLS, control_ids=(), action_ids=()),)
    )
    found = layout.patch_panel("does_not_exist", control_ids=("x",))
    assert found is False
    assert layout.panel("ctrl").control_ids == ()


# ---------------------------------------------------------------------------
# LayoutSpec.replace_panels
# ---------------------------------------------------------------------------


def test_replace_panels_swaps_full_inventory():
    layout = LayoutSpec(
        panels=(PanelSpec(id="old", kind=PANEL_KIND_CONTROLS, control_ids=("a",), action_ids=()),)
    )
    new_panels = (
        PanelSpec(id="new-ctrl", kind=PANEL_KIND_CONTROLS, control_ids=("b", "c"), action_ids=()),
    )
    layout.replace_panels(new_panels)
    assert layout.panel("old") is None
    assert layout.panel("new-ctrl") is not None
    assert layout.panel("new-ctrl").control_ids == ("b", "c")


def test_replace_panels_sets_panel_grid():
    layout = LayoutSpec(panels=())
    new_panels = (
        PanelSpec(id="p1", kind=PANEL_KIND_LINE_PLOT, view_ids=("v",)),
        PanelSpec(id="p2", kind=PANEL_KIND_CONTROLS, control_ids=(), action_ids=()),
    )
    grid = (("p1",), ("p2",))
    layout.replace_panels(new_panels, panel_grid=grid)
    assert layout.panel_grid == grid


def test_replace_panels_clears_grid_to_empty_tuple():
    layout = LayoutSpec(panels=(), panel_grid=(("p1",),))
    layout.replace_panels((), panel_grid=())
    assert layout.panel_grid == ()


# ---------------------------------------------------------------------------
# PanelPatch + LayoutReplace construction
# ---------------------------------------------------------------------------


def test_panel_patch_defaults_all_fields_to_none():
    patch = PanelPatch(panel_id="ctrl")
    assert patch.control_ids is None
    assert patch.action_ids is None
    assert patch.view_ids is None
    assert patch.title is None


def test_panel_patch_carries_explicit_values():
    patch = PanelPatch(panel_id="ctrl", control_ids=("a", "b"), action_ids=("reset",))
    assert patch.control_ids == ("a", "b")
    assert patch.action_ids == ("reset",)
    assert patch.view_ids is None


def test_layout_replace_carries_panels_and_grid():
    panels = (PanelSpec(id="p", kind=PANEL_KIND_CONTROLS, control_ids=(), action_ids=()),)
    grid = (("p",),)
    update = LayoutReplace(panels=panels, panel_grid=grid)
    assert update.panels == panels
    assert update.panel_grid == grid


def test_layout_replace_defaults_grid_to_empty():
    panels = (PanelSpec(id="p", kind=PANEL_KIND_CONTROLS, control_ids=(), action_ids=()),)
    update = LayoutReplace(panels=panels)
    assert update.panel_grid == ()
