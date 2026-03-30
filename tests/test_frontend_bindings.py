import io
import os
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
import pytest

from compneurovis import ActionSpec, AppSpec, ControlSpec, Document, Field, LayoutSpec, LinePlotViewSpec, MorphologyGeometry, MorphologyViewSpec, StateBinding, SurfaceViewSpec, VispyFrontendWindow, build_neuron_app, build_surface_app, grid_field
from compneurovis.backends.neuron import NeuronSession
from compneurovis.backends.neuron.document import NeuronDocumentBuilder
from compneurovis.frontends.vispy import frontend as frontend_module
from compneurovis.frontends.vispy.frontend import RefreshPlanner, RefreshTarget
from compneurovis.frontends.vispy.panels import LinePlotPanel, Viewport3DPanel
from compneurovis.frontends.vispy.renderers import MorphologyRenderer
from compneurovis.session import Error, FieldAppend, StatePatch, resolve_interaction_target_source


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


def test_frontend_shows_modal_for_fatal_session_error():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    class FakeTransport:
        def __init__(self):
            self._dead = True
            self.calls = 0

        def poll_updates(self):
            if self.calls == 0:
                self.calls += 1
                return [Error("Traceback (most recent call last):\nRuntimeError: boom")]
            return []

        def stop(self):
            return None

    window = VispyFrontendWindow(AppSpec(title="Error test"))
    window.timer.stop()
    window.transport = FakeTransport()

    original_critical = frontend_module.QtWidgets.QMessageBox.critical
    original_stderr = frontend_module.sys.stderr
    frontend_module.QtWidgets.QMessageBox.critical = Mock()
    frontend_module.sys.stderr = io.StringIO()
    try:
        window._poll_transport()

        frontend_module.QtWidgets.QMessageBox.critical.assert_called_once()
        args = frontend_module.QtWidgets.QMessageBox.critical.call_args[0]
        assert args[1] == "Session error"
        assert "RuntimeError: boom" in args[2]
        assert "RuntimeError: boom" in frontend_module.sys.stderr.getvalue()
        assert window.transport is None
        assert not window.timer.isActive()
    finally:
        frontend_module.QtWidgets.QMessageBox.critical = original_critical
        frontend_module.sys.stderr = original_stderr
        window.close()
        app.quit()


def test_frontend_logs_nonfatal_session_error_to_stderr_without_modal():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    class FakeTransport:
        def __init__(self):
            self._dead = False
            self.calls = 0

        def poll_updates(self):
            if self.calls == 0:
                self.calls += 1
                return [Error("Nonfatal session warning")]
            return []

        def stop(self):
            return None

    window = VispyFrontendWindow(AppSpec(title="Nonfatal error test"))
    window.timer.stop()
    window.transport = FakeTransport()

    original_critical = frontend_module.QtWidgets.QMessageBox.critical
    original_stderr = frontend_module.sys.stderr
    frontend_module.QtWidgets.QMessageBox.critical = Mock()
    frontend_module.sys.stderr = io.StringIO()
    try:
        window._poll_transport()

        frontend_module.QtWidgets.QMessageBox.critical.assert_not_called()
        assert "Nonfatal session warning" in frontend_module.sys.stderr.getvalue()
        assert window.transport is not None
        assert window.statusBar().currentMessage() == "Nonfatal session warning"
    finally:
        frontend_module.QtWidgets.QMessageBox.critical = original_critical
        frontend_module.sys.stderr = original_stderr
        window.close()
        app.quit()


def test_build_neuron_app_accepts_session_class():
    class DummyNeuronSession(NeuronSession):
        def build_sections(self):
            return []

        def initialize(self):
            raise AssertionError("should not initialize during app construction")

        def advance(self) -> None:
            return None

        def handle(self, command) -> None:
            return None

    app_spec = build_neuron_app(DummyNeuronSession, title="Dummy")

    assert app_spec.session is DummyNeuronSession
    assert app_spec.interaction_target is None
    assert app_spec.title == "Dummy"


def test_build_neuron_app_requires_lazy_session_source():
    class DefaultInteractionSession(NeuronSession):
        def __init__(self):
            super().__init__(title="Default Interaction")

        def build_sections(self):
            return []

        def initialize(self):
            raise AssertionError("should not initialize during app construction")

        def advance(self) -> None:
            return None

        def handle(self, command) -> None:
            return None

    with pytest.raises(TypeError, match="requires a Session subclass or top-level zero-argument factory"):
        build_neuron_app(DefaultInteractionSession())


def test_build_neuron_app_supports_explicit_interaction_target_factory():
    class DefaultInteractionSession(NeuronSession):
        def __init__(self):
            super().__init__(title="Default Interaction")

        def build_sections(self):
            return []

        def initialize(self):
            raise AssertionError("should not initialize during app construction")

        def advance(self) -> None:
            return None

        def handle(self, command) -> None:
            return None

    class DummyInteractionTarget:
        def __init__(self):
            self.marker = "frontend-target"

    app_spec = build_neuron_app(DefaultInteractionSession, interaction_target=DummyInteractionTarget)

    assert app_spec.interaction_target is DummyInteractionTarget
    resolved = resolve_interaction_target_source(app_spec.interaction_target)
    assert isinstance(resolved, DummyInteractionTarget)
    assert resolved.marker == "frontend-target"


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


def test_frontend_applies_state_patch_and_refreshes_bound_plot():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    field = Field(
        id="voltage",
        values=np.array([[1.0, 2.0], [10.0, 20.0]], dtype=np.float32),
        dims=("segment", "time"),
        coords={
            "segment": np.array(["seg-a", "seg-b"]),
            "time": np.array([0.0, 1.0], dtype=np.float32),
        },
    )
    view = LinePlotViewSpec(
        id="trace",
        field_id="voltage",
        x_dim="time",
        selectors={"segment": StateBinding("selected_trace_entity_ids")},
        series_dim="segment",
    )
    document = Document(
        fields={"voltage": field},
        geometries={},
        views={"trace": view},
        layout=LayoutSpec(title="State patch test", line_plot_view_id="trace"),
    )

    class FakeTransport:
        _dead = False

        def poll_updates(self):
            updates = [StatePatch({"selected_trace_entity_ids": ["seg-b"]})]
            self.poll_updates = lambda: []
            return updates

        def stop(self):
            return None

    window = VispyFrontendWindow(AppSpec(document=document, title="State patch test"))
    window.timer.stop()
    window.transport = FakeTransport()

    window._poll_transport()

    assert window.state["selected_trace_entity_ids"] == ["seg-b"]
    assert "seg-b" in window.line_plot._series_items
    window.close()
    app.quit()


def test_line_plot_panel_uses_series_palette_for_multi_series_colors():
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
        series_palette=("#ff0000", "#0000ff"),
    )

    panel.refresh(view, field, {}, {})

    ligand_pen = panel._series_items["ligand"].opts["pen"]
    receptor_pen = panel._series_items["receptor"].opts["pen"]
    assert ligand_pen.color().name() == "#ff0000"
    assert receptor_pen.color().name() == "#0000ff"
    app.quit()


def test_line_plot_panel_applies_rolling_window_and_y_range():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = LinePlotPanel()
    field = Field(
        id="trace",
        values=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        dims=("time",),
        coords={"time": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)},
    )
    view = LinePlotViewSpec(
        id="trace-plot",
        field_id=field.id,
        x_dim="time",
        rolling_window=1.5,
        y_min=0.0,
        y_max=5.0,
    )

    panel.refresh(view, field, {}, {})
    view_range = panel.plotItem.getViewBox().viewRange()

    assert np.allclose(view_range[0], [1.5, 3.0])
    assert np.allclose(view_range[1], [0.0, 5.0])
    app.quit()


def test_line_plot_panel_clamps_rolling_window_to_available_history():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = LinePlotPanel()
    field = Field(
        id="trace",
        values=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        dims=("time",),
        coords={"time": np.array([0.0, 50.0, 100.0], dtype=np.float32)},
    )
    view = LinePlotViewSpec(
        id="trace-plot",
        field_id=field.id,
        x_dim="time",
        rolling_window=120.0,
    )

    panel.refresh(view, field, {}, {})
    view_range = panel.plotItem.getViewBox().viewRange()

    assert np.allclose(view_range[0], [0.0, 100.0])
    app.quit()


def test_line_plot_panel_resets_single_sample_rolling_window_without_negative_backfill():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = LinePlotPanel()
    view = LinePlotViewSpec(
        id="trace-plot",
        field_id="trace",
        x_dim="time",
        rolling_window=30.0,
    )
    field_before = Field(
        id="trace",
        values=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        dims=("time",),
        coords={"time": np.array([90.0, 100.0, 120.0], dtype=np.float32)},
    )
    field_after = Field(
        id="trace",
        values=np.array([1.0], dtype=np.float32),
        dims=("time",),
        coords={"time": np.array([0.0], dtype=np.float32)},
    )

    panel.refresh(view, field_before, {}, {})
    panel.refresh(view, field_after, {}, {})
    view_range = panel.plotItem.getViewBox().viewRange()

    assert np.allclose(view_range[0], [0.0, 30.0])
    app.quit()


def test_line_plot_panel_can_trim_data_to_rolling_window():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = LinePlotPanel()
    field = Field(
        id="trace",
        values=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        dims=("time",),
        coords={"time": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)},
    )
    view = LinePlotViewSpec(
        id="trace-plot",
        field_id=field.id,
        x_dim="time",
        rolling_window=1.5,
        trim_to_rolling_window=True,
    )

    panel.refresh(view, field, {}, {})
    x_data, y_data = panel._plot_item.getData()

    assert np.allclose(x_data, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert np.allclose(y_data, np.array([2.0, 3.0, 4.0], dtype=np.float32))
    app.quit()


def test_multi_series_line_plot_can_trim_data_to_rolling_window():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = LinePlotPanel()
    field = Field(
        id="cascade",
        values=np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], dtype=np.float32),
        dims=("series", "time"),
        coords={
            "series": np.array(["ligand", "receptor"]),
            "time": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        },
    )
    view = LinePlotViewSpec(
        id="cascade-plot",
        field_id=field.id,
        x_dim="time",
        series_dim="series",
        rolling_window=1.5,
        trim_to_rolling_window=True,
    )

    panel.refresh(view, field, {}, {})
    ligand_x, ligand_y = panel._series_items["ligand"].getData()
    receptor_x, receptor_y = panel._series_items["receptor"].getData()

    assert np.allclose(ligand_x, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert np.allclose(ligand_y, np.array([2.0, 3.0, 4.0], dtype=np.float32))
    assert np.allclose(receptor_x, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert np.allclose(receptor_y, np.array([20.0, 30.0, 40.0], dtype=np.float32))
    app.quit()


def test_line_plot_trim_keeps_last_sample_before_window_boundary():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = LinePlotPanel()
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    view = LinePlotViewSpec(
        id="trace-plot",
        field_id="trace",
        x_dim="time",
        rolling_window=1.5,
        trim_to_rolling_window=True,
    )

    trimmed_x, trimmed_y = panel._trim_line_data(view, x, y)

    assert np.allclose(trimmed_x, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert np.allclose(trimmed_y, np.array([2.0, 3.0, 4.0], dtype=np.float32))
    app.quit()


def test_line_plot_trim_drops_nonfinite_history_samples():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = LinePlotPanel()
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([np.nan, np.nan, 3.0, 4.0], dtype=np.float32)
    view = LinePlotViewSpec(
        id="trace-plot",
        field_id="trace",
        x_dim="time",
        rolling_window=10.0,
        trim_to_rolling_window=True,
    )

    trimmed_x, trimmed_y = panel._trim_line_data(view, x, y)

    assert np.allclose(trimmed_x, np.array([2.0, 3.0], dtype=np.float32))
    assert np.allclose(trimmed_y, np.array([3.0, 4.0], dtype=np.float32))
    app.quit()


def test_line_plot_panel_supports_multi_selected_morphology_traces():
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
    view = LinePlotViewSpec(
        id="trace",
        field_id=field.id,
        x_dim="time",
        series_dim="segment",
        selectors={"segment": StateBinding("selected_trace_entity_ids")},
        title="Selected voltages",
    )

    panel.refresh(view, field, {"selected_trace_entity_ids": ["seg-b", "seg-a"]}, {})

    assert set(panel._series_items.keys()) == {"seg-a", "seg-b"}
    seg_b_x, seg_b_y = panel._series_items["seg-b"].getData()
    assert np.allclose(seg_b_x, np.array([0.0, 1.0, 2.0], dtype=np.float32))
    assert np.allclose(seg_b_y, np.array([10.0, 20.0, 30.0], dtype=np.float32))
    app.quit()


def test_line_plot_panel_drops_nonfinite_prefix_for_sparse_selected_trace_history():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = LinePlotPanel()
    field = Field(
        id="voltage_trace",
        values=np.array([[1.0, 2.0, 3.0, 4.0], [np.nan, np.nan, 30.0, 40.0]], dtype=np.float32),
        dims=("segment", "time"),
        coords={
            "segment": np.array(["seg-a", "seg-b"]),
            "time": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        },
    )
    view = LinePlotViewSpec(
        id="trace",
        field_id=field.id,
        x_dim="time",
        series_dim="segment",
        selectors={"segment": StateBinding("selected_trace_entity_ids")},
        trim_to_rolling_window=True,
        rolling_window=10.0,
    )

    panel.refresh(view, field, {"selected_trace_entity_ids": ["seg-a", "seg-b"]}, {})

    seg_b_x, seg_b_y = panel._series_items["seg-b"].getData()
    assert np.allclose(seg_b_x, np.array([2.0, 3.0], dtype=np.float32))
    assert np.allclose(seg_b_y, np.array([30.0, 40.0], dtype=np.float32))
    app.quit()


def test_line_plot_panel_ignores_missing_selected_trace_labels_for_sparse_history():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = LinePlotPanel()
    field = Field(
        id="voltage_trace",
        values=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        dims=("segment", "time"),
        coords={
            "segment": np.array(["seg-a"]),
            "time": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        },
    )
    view = LinePlotViewSpec(
        id="trace",
        field_id=field.id,
        x_dim="time",
        series_dim="segment",
        selectors={"segment": StateBinding("selected_trace_entity_ids")},
        title="Selected voltages",
    )

    panel.refresh(view, field, {"selected_trace_entity_ids": ["seg-a", "seg-b"]}, {})

    assert set(panel._series_items.keys()) == {"seg-a"}
    app.quit()


def test_line_plot_panel_applies_explicit_x_tick_spacing():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = LinePlotPanel()
    field = Field(
        id="trace",
        values=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        dims=("time",),
        coords={"time": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)},
    )
    view = LinePlotViewSpec(
        id="trace-plot",
        field_id=field.id,
        x_dim="time",
        x_major_tick_spacing=5.0,
        x_minor_tick_spacing=1.0,
        rolling_window=30.0,
    )
    panel.refresh(view, field, {}, {})
    ticks = panel.plotItem.getAxis("bottom")._tickLevels
    assert ticks is not None
    major = ticks[0]
    minor = ticks[1]
    assert [value for value, _ in major] == [0.0]
    assert [label for _, label in major] == ["0"]
    assert [value for value, _ in minor] == [1.0, 2.0, 3.0]
    assert all(label == "" for _, label in minor)
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

    assert window.viewport.isHidden()
    assert isinstance(window.centralWidget(), QtWidgets.QSplitter)
    assert window._horizontal_splitter.orientation() == QtCore.Qt.Orientation.Horizontal
    assert window._right_splitter.orientation() == QtCore.Qt.Orientation.Vertical

    window.close()
    app.quit()


def test_frontend_uses_splitters_for_draggable_panel_resize():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = VispyFrontendWindow(build_surface_cross_section_app())
    window.timer.stop()

    assert isinstance(window.centralWidget(), QtWidgets.QSplitter)
    assert window._horizontal_splitter.widget(0) is window.viewport
    assert window._horizontal_splitter.widget(1) is window._right_splitter
    assert window._right_splitter.widget(0) is window.line_plot
    assert window._right_splitter.widget(1) is window.controls
    assert not window._horizontal_splitter.opaqueResize()
    assert not window._right_splitter.opaqueResize()

    window.close()
    app.quit()


def test_controls_panel_renders_and_dispatches_document_actions():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    field = Field(
        id="voltage",
        values=np.array([[1.0], [2.0]], dtype=np.float32),
        dims=("segment", "time"),
        coords={
            "segment": np.array(["seg-a", "seg-b"]),
            "time": np.array([0.0], dtype=np.float32),
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
    document = Document(
        fields={field.id: field},
        geometries={geometry.id: geometry},
        views={
            "morphology": MorphologyViewSpec(id="morphology", geometry_id=geometry.id)
        },
        actions={
            "mark_selected": ActionSpec(
                id="mark_selected",
                label="Mark selected",
                payload={"entity_id": StateBinding("selected_entity_id")},
            )
        },
        layout=LayoutSpec(title="Action test", main_3d_view_id="morphology", action_ids=("mark_selected",)),
    )
    window = VispyFrontendWindow(AppSpec(document=document, title="Action test"))
    window.timer.stop()
    window.transport = Mock()

    action_button = window.controls.widgets["mark_selected"]
    action_button.click()

    window.transport.send_command.assert_called_once()
    command = window.transport.send_command.call_args[0][0]
    assert command.action_id == "mark_selected"
    assert command.payload["entity_id"] == "seg-a"

    window.close()
    app.quit()


def test_action_shortcut_dispatches_invoke_action():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    document = Document(
        fields={},
        geometries={},
        views={},
        actions={
            "toggle": ActionSpec(
                id="toggle",
                label="Toggle",
                payload={"selected": StateBinding("selected_entity_id")},
                shortcuts=("I",),
            )
        },
        layout=LayoutSpec(title="Shortcut test", action_ids=("toggle",)),
    )
    window = VispyFrontendWindow(AppSpec(document=document, title="Shortcut test"))
    window.timer.stop()
    window.transport = Mock()
    window.state["selected_entity_id"] = "seg-a"

    event = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_I, QtCore.Qt.KeyboardModifier.NoModifier, "i")
    window.keyPressEvent(event)

    window.transport.send_command.assert_called_once()
    command = window.transport.send_command.call_args[0][0]
    assert command.action_id == "toggle"
    assert command.payload["selected"] == "seg-a"

    window.close()
    app.quit()


def test_selection_mode_action_arms_on_shortcut_and_fires_on_click():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    field = Field(
        id="voltage",
        values=np.array([[1.0], [2.0]], dtype=np.float32),
        dims=("segment", "time"),
        coords={
            "segment": np.array(["seg-a", "seg-b"]),
            "time": np.array([0.0], dtype=np.float32),
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
    document = Document(
        fields={field.id: field},
        geometries={geometry.id: geometry},
        views={"morphology": MorphologyViewSpec(id="morphology", geometry_id=geometry.id)},
        actions={
            "arm_add": ActionSpec(
                id="arm_add",
                label="Arm add",
                shortcuts=("1",),
                selection_mode=True,
            )
        },
        layout=LayoutSpec(title="Selection action test", main_3d_view_id="morphology", action_ids=("arm_add",)),
    )
    window = VispyFrontendWindow(AppSpec(document=document, title="Selection action test"))
    window.timer.stop()
    window.transport = Mock()

    arm_event = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_1, QtCore.Qt.KeyboardModifier.NoModifier, "1")
    window.keyPressEvent(arm_event)
    assert window.transport.send_command.call_count == 0

    window._on_entity_selected("seg-b")

    window.transport.send_command.assert_called_once()
    command = window.transport.send_command.call_args[0][0]
    assert command.action_id == "arm_add"
    assert command.payload["entity_id"] == "seg-b"

    window.close()
    app.quit()


class DummyInteractionTarget:
    def __init__(self):
        self.mode_enabled = False
        self.clicked_entities = []

    def on_action(self, action_id, payload, context):
        del payload
        if action_id == "toggle_mode":
            self.mode_enabled = not self.mode_enabled
            context.show_status("Mode enabled" if self.mode_enabled else "Mode disabled")
            return True
        if action_id == "clear_targets":
            self.clicked_entities.clear()
            return False
        return False

    def on_entity_clicked(self, entity_id, context):
        if not self.mode_enabled:
            return False
        self.clicked_entities.append(entity_id)
        context.invoke_action("register_entity", {"entity_id": entity_id})
        return True


def test_interaction_callbacks_can_handle_mode_and_entity_click_without_selection_mode():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    field = Field(
        id="voltage",
        values=np.array([[1.0], [2.0]], dtype=np.float32),
        dims=("segment", "time"),
        coords={
            "segment": np.array(["seg-a", "seg-b"]),
            "time": np.array([0.0], dtype=np.float32),
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
    document = Document(
        fields={field.id: field},
        geometries={geometry.id: geometry},
        views={"morphology": MorphologyViewSpec(id="morphology", geometry_id=geometry.id)},
        actions={
            "toggle_mode": ActionSpec(id="toggle_mode", label="Toggle mode", shortcuts=("1",)),
            "register_entity": ActionSpec(id="register_entity", label="Register entity"),
            "clear_targets": ActionSpec(id="clear_targets", label="Clear targets"),
        },
        layout=LayoutSpec(
            title="Callback interaction test",
            main_3d_view_id="morphology",
            action_ids=("toggle_mode", "clear_targets"),
        ),
    )
    interaction_target = DummyInteractionTarget()
    window = VispyFrontendWindow(AppSpec(document=document, interaction_target=interaction_target, title="Callback interaction test"))
    window.timer.stop()
    window.transport = Mock()

    toggle_event = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_1, QtCore.Qt.KeyboardModifier.NoModifier, "1")
    window.keyPressEvent(toggle_event)
    assert interaction_target.mode_enabled is True
    assert window.transport.send_command.call_count == 0

    window._on_entity_selected("seg-b")

    window.transport.send_command.assert_called_once()
    command = window.transport.send_command.call_args[0][0]
    assert command.action_id == "register_entity"
    assert command.payload["entity_id"] == "seg-b"
    assert interaction_target.clicked_entities == ["seg-b"]

    window.close()
    app.quit()


def test_frontend_interaction_context_resolves_entity_info_from_document():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    field = Field(
        id="voltage",
        values=np.array([[1.0], [2.0]], dtype=np.float32),
        dims=("segment", "time"),
        coords={
            "segment": np.array(["seg-a", "seg-b"]),
            "time": np.array([0.0], dtype=np.float32),
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
    document = Document(
        fields={field.id: field},
        geometries={geometry.id: geometry},
        views={"morphology": MorphologyViewSpec(id="morphology", geometry_id=geometry.id)},
        layout=LayoutSpec(title="Entity info test", main_3d_view_id="morphology"),
    )
    window = VispyFrontendWindow(AppSpec(document=document, title="Entity info test"))
    window.timer.stop()

    info = window._interaction_context().entity_info("seg-b")

    assert info is not None
    assert info["entity_id"] == "seg-b"
    assert info["section_name"] == "sec-b"
    assert info["label"] == "sec-b@0.9"

    window.close()
    app.quit()


class DummyNeuronSession(NeuronSession):
    def __init__(self):
        super().__init__(title="Dummy neuron app")

    def build_sections(self):
        return []

    def control_specs(self):
        return {"gain": ControlSpec("gain", "float", "Gain", 1.0)}

    def action_specs(self):
        return {"toggle_mode": ActionSpec("toggle_mode", "Toggle mode")}

    def control_order(self):
        return ("gain",)

    def action_order(self):
        return ("toggle_mode",)

    def trace_view_updates(self):
        return {"rolling_window": 25.0, "trim_to_rolling_window": True}


def test_neuron_session_build_document_applies_orders_and_trace_updates():
    session = DummyNeuronSession()
    geometry = MorphologyGeometry(
        id="morphology",
        positions=np.zeros((1, 3), dtype=np.float32),
        orientations=np.eye(3, dtype=np.float32)[None, :, :],
        radii=np.ones(1, dtype=np.float32),
        lengths=np.ones(1, dtype=np.float32),
        entity_ids=("seg-a",),
        section_names=("sec-a",),
        xlocs=np.array([0.5], dtype=np.float32),
        labels=("sec-a@0.5",),
    )

    document = session.build_document(
        geometry=geometry,
        voltage_values=np.array([1.0], dtype=np.float32),
        time_value=0.0,
    )

    assert document.layout.control_ids == ("gain",)
    assert document.layout.action_ids == ("toggle_mode",)
    trace_view = document.views["trace"]
    assert trace_view.rolling_window == 25.0
    assert trace_view.trim_to_rolling_window is True


def test_neuron_session_defaults_to_batched_display_updates():
    session = DummyNeuronSession()

    assert session.steps_per_update() == 5


class WindowBudgetNeuronSession(NeuronSession):
    def __init__(self):
        super().__init__(dt=0.1, max_samples=1000, title="Window budget")

    def build_sections(self):
        return []

    def trace_view_updates(self):
        return {"rolling_window": 120.0}


def test_neuron_session_expands_history_budget_to_cover_trace_window():
    session = WindowBudgetNeuronSession()
    geometry = MorphologyGeometry(
        id="morphology",
        positions=np.zeros((1, 3), dtype=np.float32),
        orientations=np.eye(3, dtype=np.float32)[None, :, :],
        radii=np.ones(1, dtype=np.float32),
        lengths=np.ones(1, dtype=np.float32),
        entity_ids=("seg-a",),
        section_names=("sec-a",),
        xlocs=np.array([0.5], dtype=np.float32),
        labels=("sec-a@0.5",),
    )
    document = session.build_document(
        geometry=geometry,
        voltage_values=np.array([1.0], dtype=np.float32),
        time_value=0.0,
    )

    assert session._resolved_field_max_samples(document, field_id=NeuronDocumentBuilder.TRACE_FIELD_ID, append_dim="time") == 1201


def test_frontend_applies_field_append_updates_incrementally():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    field = Field(
        id="voltage",
        values=np.array([[1.0], [10.0]], dtype=np.float32),
        dims=("segment", "time"),
        coords={
            "segment": np.array(["seg-a", "seg-b"]),
            "time": np.array([0.0], dtype=np.float32),
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
    document = Document(
        fields={field.id: field},
        geometries={geometry.id: geometry},
        views={
            "morphology": MorphologyViewSpec(
                id="morphology",
                geometry_id=geometry.id,
                color_field_id=field.id,
                entity_dim="segment",
                sample_dim="time",
            ),
            "trace": LinePlotViewSpec(
                id="trace",
                field_id=field.id,
                x_dim="time",
                selectors={"segment": StateBinding("selected_entity_id")},
            ),
        },
        layout=LayoutSpec(title="Append test", main_3d_view_id="morphology", line_plot_view_id="trace"),
    )
    window = VispyFrontendWindow(AppSpec(document=document, title="Append test"))
    window.timer.stop()
    window.transport = Mock()
    window.transport.poll_updates.return_value = [
        FieldAppend(
            field_id="voltage",
            append_dim="time",
            values=np.array([[2.0, 3.0], [20.0, 30.0]], dtype=np.float32),
            coord_values=np.array([1.0, 2.0], dtype=np.float32),
            max_length=2,
        )
    ]

    window._poll_transport()

    updated = window.document.fields["voltage"]
    assert updated.coord("time").tolist() == [1.0, 2.0]
    assert np.allclose(updated.values, np.array([[2.0, 3.0], [20.0, 30.0]], dtype=np.float32))
    x_data, y_data = window.line_plot._plot_item.getData()
    assert np.allclose(x_data, np.array([1.0, 2.0], dtype=np.float32))
    assert np.allclose(y_data, np.array([2.0, 3.0], dtype=np.float32))

    window.close()
    app.quit()


def test_frontend_batches_multiple_field_appends_into_one_refresh_pass():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    field = Field(
        id="voltage",
        values=np.array([[1.0], [10.0]], dtype=np.float32),
        dims=("segment", "time"),
        coords={
            "segment": np.array(["seg-a", "seg-b"]),
            "time": np.array([0.0], dtype=np.float32),
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
    document = Document(
        fields={field.id: field},
        geometries={geometry.id: geometry},
        views={
            "morphology": MorphologyViewSpec(
                id="morphology",
                geometry_id=geometry.id,
                color_field_id=field.id,
                entity_dim="segment",
                sample_dim="time",
            ),
            "trace": LinePlotViewSpec(
                id="trace",
                field_id=field.id,
                x_dim="time",
                selectors={"segment": StateBinding("selected_entity_id")},
            ),
        },
        layout=LayoutSpec(title="Batch append test", main_3d_view_id="morphology", line_plot_view_id="trace"),
    )
    window = VispyFrontendWindow(AppSpec(document=document, title="Batch append test"))
    window.timer.stop()
    window.transport = Mock()
    window._refresh_morphology = Mock()
    window._refresh_line_plot = Mock()
    window.viewport.commit = Mock()
    window.transport.poll_updates.return_value = [
        FieldAppend(
            field_id="voltage",
            append_dim="time",
            values=np.array([[2.0], [20.0]], dtype=np.float32),
            coord_values=np.array([1.0], dtype=np.float32),
            max_length=3,
        ),
        FieldAppend(
            field_id="voltage",
            append_dim="time",
            values=np.array([[3.0], [30.0]], dtype=np.float32),
            coord_values=np.array([2.0], dtype=np.float32),
            max_length=3,
        ),
    ]

    window._poll_transport()

    updated = window.document.fields["voltage"]
    assert updated.coord("time").tolist() == [0.0, 1.0, 2.0]
    assert np.allclose(updated.values, np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32))
    window._refresh_morphology.assert_called_once()
    window._refresh_line_plot.assert_called_once()
    window.viewport.commit.assert_called_once()

    window.close()
    app.quit()


def test_multi_series_refresh_keeps_existing_legend_when_series_are_stable():
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
        show_legend=True,
    )

    panel.refresh(view, field, {}, {})
    legend = panel.plotItem.legend
    assert legend is not None
    original_clear = legend.clear
    original_add = legend.addItem
    legend.clear = Mock(wraps=original_clear)
    legend.addItem = Mock(wraps=original_add)

    panel.refresh(view, field.with_values(field.values + 1.0), {}, {})

    legend.clear.assert_not_called()
    legend.addItem.assert_not_called()
    app.quit()


def test_morphology_renderer_decode_pick_index_reads_exact_pick_pixel():
    renderer = MorphologyRenderer.__new__(MorphologyRenderer)
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    cid = 2
    img[0, 0, 0] = cid & 0xFF
    img[0, 0, 1] = (cid >> 8) & 0xFF
    img[0, 0, 2] = (cid >> 16) & 0xFF

    idx = renderer._decode_pick_index(img)

    assert idx == 1
