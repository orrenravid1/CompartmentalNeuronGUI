import os
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6 import QtWidgets

from compneurovis import AppSpec, ControlSpec, Document, Field, LayoutSpec, LinePlotViewSpec, MorphologyGeometry, StateBinding, SurfaceViewSpec, VispyFrontendWindow, build_surface_app, grid_field
from compneurovis.frontends.vispy import frontend as frontend_module
from compneurovis.frontends.vispy.frontend import RefreshPlanner, RefreshTarget
from compneurovis.frontends.vispy.panels import LinePlotPanel, Viewport3DPanel


def test_line_plot_panel_resolves_selected_entity_binding():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = LinePlotPanel()
    field = Field(
        id="voltage",
        values=np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32),
        dims=("segment", "time"),
        coords={
            "segment": np.array(["seg-a", "seg-b"]),
            "time": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        },
    )
    geometry = MorphologyGeometry(
        id="morphology",
        positions=np.zeros((2, 3), dtype=np.float32),
        orientations=np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0),
        radii=np.ones(2, dtype=np.float32),
        lengths=np.ones(2, dtype=np.float32),
        entity_ids=("seg-a", "seg-b"),
        section_names=("sec-a", "sec-b"),
        xlocs=np.array([0.1, 0.9], dtype=np.float32),
        labels=("sec-a@0.1", "sec-b@0.9"),
    )
    view = LinePlotViewSpec(
        id="trace",
        field_id=field.id,
        x_dim="time",
        selectors={"segment": StateBinding("selected_entity_id")},
        title="Voltage",
        x_label="Time",
        y_label="Voltage",
    )

    panel.refresh(view, field, {"selected_entity_id": "seg-b"}, {"morphology": geometry})
    x_data, y_data = panel._plot_item.getData()
    assert np.allclose(x_data, np.array([0.0, 1.0, 2.0], dtype=np.float32))
    assert np.allclose(y_data, np.array([10.0, 20.0, 30.0], dtype=np.float32))
    assert "sec-b@0.9" in panel.plotItem.titleLabel.text
    app.quit()


def build_surface_cross_section_app():
    x = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    y = np.linspace(-2.0, 2.0, 6, dtype=np.float32)
    values = (np.sin(x[None, :]) + np.cos(y[:, None])).astype(np.float32)
    field, geometry = grid_field(field_id="surface", values=values, x_coords=x, y_coords=y)
    controls = {
        "slice_axis": ControlSpec("slice_axis", "enum", "Slice axis", "x", options=("x", "y")),
        "slice_position": ControlSpec("slice_position", "float", "Slice position", 0.0, min=0.0, max=1.0, steps=20),
    }
    surface_view = SurfaceViewSpec(
        id="surface-view",
        field_id=field.id,
        geometry_id=geometry.id,
        slice_axis_state_key="slice_axis",
        slice_position_state_key="slice_position",
    )
    line_view = LinePlotViewSpec(
        id="surface-line",
        field_id=field.id,
        orthogonal_slice_state_key="slice_axis",
        orthogonal_position_state_key="slice_position",
    )
    return build_surface_app(
        field=field,
        geometry=geometry,
        title="surface test",
        surface_view=surface_view,
        line_view=line_view,
        controls=controls,
    )


def test_refresh_planner_targets_slice_state_to_overlay_and_line_plot():
    app_spec = build_surface_cross_section_app()
    planner = RefreshPlanner(app_spec.document)

    assert planner.targets_for_state_change("slice_position") == {
        RefreshTarget.SURFACE_SLICE,
        RefreshTarget.LINE_PLOT,
    }
    assert planner.targets_for_state_change("slice_axis") == {
        RefreshTarget.SURFACE_SLICE,
        RefreshTarget.LINE_PLOT,
    }


def test_surface_control_change_avoids_surface_mesh_refresh():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = VispyFrontendWindow(build_surface_cross_section_app())
    window.timer.stop()
    window.viewport.refresh_surface_visual = Mock()
    window.viewport.refresh_surface_axes = Mock()
    window.viewport.refresh_surface_slice = Mock()
    window.viewport.commit = Mock()
    window.line_plot.refresh = Mock()

    window._on_control_changed(window.document.controls["slice_position"], 0.4)

    window.viewport.refresh_surface_visual.assert_not_called()
    window.viewport.refresh_surface_axes.assert_not_called()
    window.viewport.refresh_surface_slice.assert_called_once()
    window.viewport.commit.assert_called_once()
    window.line_plot.refresh.assert_called_once()

    window.close()
    app.quit()


def test_surface_slice_overlay_moves_with_control_changes():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = VispyFrontendWindow(build_surface_cross_section_app())
    window.timer.stop()

    initial_pos = np.array(window.viewport.renderer_surface.slice_overlay._line._pos, copy=True)
    window._on_control_changed(window.document.controls["slice_position"], 1.0)
    moved_pos = np.array(window.viewport.renderer_surface.slice_overlay._line._pos, copy=True)
    window._on_control_changed(window.document.controls["slice_axis"], "y")
    axis_switched_pos = np.array(window.viewport.renderer_surface.slice_overlay._line._pos, copy=True)

    assert not np.allclose(initial_pos, moved_pos)
    assert not np.allclose(moved_pos, axis_switched_pos)

    window.close()
    app.quit()


def test_surface_visual_reuses_same_surface_object_for_same_shape_updates():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = Viewport3DPanel()
    x = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    y = np.linspace(-2.0, 2.0, 6, dtype=np.float32)
    values = (np.sin(x[None, :]) + np.cos(y[:, None])).astype(np.float32)
    field, geometry = grid_field(field_id="surface", values=values, x_coords=x, y_coords=y)
    view = SurfaceViewSpec(id="surface-view", field_id=field.id, geometry_id=geometry.id)
    state = {
        "surface-view:cmap": "bwr",
        "surface-view:clim": None,
        "surface-view:color_by": "height",
        "surface-view:surface_alpha": 1.0,
        "surface-view:background_color": "white",
    }

    panel.refresh_surface_visual(surface_view=view, surface_field=field, grid_geometry=geometry, resolved_state=state)
    first_surface = panel.renderer_surface.surface

    panel.refresh_surface_visual(
        surface_view=view,
        surface_field=field.with_values(field.values + 1.0),
        grid_geometry=geometry,
        resolved_state=state,
    )
    second_surface = panel.renderer_surface.surface

    assert first_surface is second_surface
    app.quit()


def test_run_app_skips_frontend_launch_in_spawned_child():
    original_current_process = frontend_module.mp.current_process
    original_qapplication = frontend_module.QtWidgets.QApplication
    try:
        frontend_module.mp.current_process = lambda: SimpleNamespace(name="SpawnProcess-1")
        frontend_module.QtWidgets.QApplication = Mock()
        frontend_module.run_app(frontend_module.AppSpec())
        frontend_module.QtWidgets.QApplication.assert_not_called()
    finally:
        frontend_module.mp.current_process = original_current_process
        frontend_module.QtWidgets.QApplication = original_qapplication


def test_line_plot_panel_supports_multi_series_fields():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = LinePlotPanel()
    field = Field(
        id="cascade",
        values=np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32),
        dims=("series", "time"),
        coords={
            "series": np.array(["ligand", "receptor"]),
            "time": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        },
    )
    view = LinePlotViewSpec(
        id="cascade-plot",
        field_id=field.id,
        x_dim="time",
        series_dim="series",
        title="Cascade",
        x_label="Time",
        y_label="Value",
        series_colors={"ligand": "#ff0000", "receptor": "#0000ff"},
    )

    panel.refresh(view, field, {}, {})

    assert set(panel._series_items.keys()) == {"ligand", "receptor"}
    ligand_x, ligand_y = panel._series_items["ligand"].getData()
    receptor_x, receptor_y = panel._series_items["receptor"].getData()
    assert np.allclose(ligand_x, np.array([0.0, 1.0, 2.0], dtype=np.float32))
    assert np.allclose(ligand_y, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert np.allclose(receptor_x, np.array([0.0, 1.0, 2.0], dtype=np.float32))
    assert np.allclose(receptor_y, np.array([10.0, 20.0, 30.0], dtype=np.float32))
    app.quit()


def test_frontend_hides_viewport_when_document_has_no_3d_view():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    field = Field(
        id="cascade",
        values=np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32),
        dims=("series", "time"),
        coords={
            "series": np.array(["ligand", "receptor"]),
            "time": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        },
    )
    document = Document(
        fields={field.id: field},
        geometries={},
        views={
            "cascade-plot": LinePlotViewSpec(
                id="cascade-plot",
                field_id=field.id,
                x_dim="time",
                series_dim="series",
            )
        },
        layout=LayoutSpec(title="Cascade", main_3d_view_id=None, line_plot_view_id="cascade-plot"),
    )
    window = VispyFrontendWindow(AppSpec(document=document, title="Cascade"))
    window.timer.stop()

    assert not window.viewport.isVisible()
    assert window._central_layout.stretch(0) == 0
    assert window._central_layout.stretch(1) == 1

    window.close()
    app.quit()
