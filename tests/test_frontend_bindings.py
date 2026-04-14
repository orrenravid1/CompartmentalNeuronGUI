import io
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
import pytest

from compneurovis import ActionSpec, AppSpec, ControlSpec, Field, GridSliceOperatorSpec, LayoutSpec, LinePlotViewSpec, MorphologyGeometry, MorphologyViewSpec, Scene, StateBinding, SurfaceViewSpec, View3DHostSpec, VispyFrontendWindow, build_neuron_app, build_surface_app, grid_field
from compneurovis.backends.neuron import NeuronSession
from compneurovis.backends.neuron.scene import NeuronSceneBuilder
from compneurovis.frontends.vispy import frontend as frontend_module
from compneurovis.frontends.vispy.frontend import RefreshPlanner, RefreshTarget
from compneurovis.frontends.vispy.panels import ControlsHostPanel, LinePlotHostPanel, LinePlotPanel, Viewport3DPanel
from compneurovis.frontends.vispy.renderers import MorphologyRenderer
from compneurovis.session import BufferedSession, Error, FieldAppend, FieldReplace, Reset, StatePatch, resolve_interaction_target_source


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
    assert "sec-b@0.9" in panel.resolved_title
    app.quit()


def test_line_plot_host_uses_resolved_plot_title():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = LinePlotHostPanel(view_id="trace", title="Trace")
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

    assert "sec-b@0.9" in panel.title()
    assert panel.line_plot_panel.plotItem.titleLabel.text == ""
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
    operator = GridSliceOperatorSpec(
        id="surface-slice",
        field_id=field.id,
        geometry_id=geometry.id,
        axis_state_key="slice_axis",
        position_state_key="slice_position",
    )
    surface_view = SurfaceViewSpec(id="surface-view", field_id=field.id, geometry_id=geometry.id)
    line_view = LinePlotViewSpec(
        id="surface-line",
        operator_id=operator.id,
    )
    return build_surface_app(
        field=field,
        geometry=geometry,
        title="surface test",
        surface_view=surface_view,
        line_views=(line_view,),
        operators={operator.id: operator},
        controls=controls,
        view_3d_hosts=(View3DHostSpec(id="surface-host", view_ids=("surface-view",), operator_ids=(operator.id,)),),
    )


def build_surface_axes_binding_app():
    x = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    y = np.linspace(-2.0, 2.0, 6, dtype=np.float32)
    values = (np.sin(x[None, :]) + np.cos(y[:, None])).astype(np.float32)
    field, geometry = grid_field(field_id="surface", values=values, x_coords=x, y_coords=y)
    controls = {
        "tick_count": ControlSpec("tick_count", "int", "Axis ticks", 7, min=0, max=12),
        "tick_label_size": ControlSpec("tick_label_size", "float", "Tick text size", 12.0, min=6.0, max=24.0, steps=18),
        "axis_label_size": ControlSpec("axis_label_size", "float", "Axis label size", 16.0, min=8.0, max=32.0, steps=24),
        "axis_color": ControlSpec("axis_color", "enum", "Axis color", "black", options=("black", "red")),
        "text_color": ControlSpec("text_color", "enum", "Text color", "black", options=("black", "blue")),
        "axis_alpha": ControlSpec("axis_alpha", "float", "Axis alpha", 1.0, min=0.0, max=1.0, steps=10),
    }
    surface_view = SurfaceViewSpec(
        id="surface-view",
        field_id=field.id,
        geometry_id=geometry.id,
        render_axes=True,
        axes_in_middle=True,
        tick_count=StateBinding("tick_count"),
        tick_label_size=StateBinding("tick_label_size"),
        axis_label_size=StateBinding("axis_label_size"),
        axis_color=StateBinding("axis_color"),
        text_color=StateBinding("text_color"),
        axis_alpha=StateBinding("axis_alpha"),
        axis_labels=("x", "y", "height"),
    )
    return build_surface_app(
        field=field,
        geometry=geometry,
        title="surface axes test",
        surface_view=surface_view,
        controls=controls,
        view_3d_hosts=(View3DHostSpec(id="surface-host", view_ids=("surface-view",)),),
    )


def test_refresh_planner_targets_slice_state_to_overlay_and_line_plot():
    app_spec = build_surface_cross_section_app()
    planner = RefreshPlanner(app_spec.scene)

    assert planner.targets_for_state_change("slice_position") == {
        RefreshTarget.operator_overlay("surface-view"),
        RefreshTarget.line_plot("surface-line"),
    }
    assert planner.targets_for_state_change("slice_axis") == {
        RefreshTarget.operator_overlay("surface-view"),
        RefreshTarget.line_plot("surface-line"),
    }


def test_refresh_planner_splits_surface_axis_geometry_and_style_targets():
    app_spec = build_surface_axes_binding_app()
    planner = RefreshPlanner(app_spec.scene)

    assert planner.targets_for_state_change("tick_count") == {
        RefreshTarget.surface_axes_geometry("surface-view"),
    }
    assert planner.targets_for_state_change("tick_label_size") == {
        RefreshTarget.surface_axes_style("surface-view"),
    }
    assert planner.targets_for_state_change("axis_color") == {
        RefreshTarget.surface_axes_style("surface-view"),
    }


def test_surface_control_change_avoids_surface_mesh_refresh():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = VispyFrontendWindow(build_surface_cross_section_app())
    window.timer.stop()
    window.viewport.refresh_surface_visual = Mock()
    window.viewport.refresh_surface_axes_geometry = Mock()
    window.viewport.refresh_surface_axes_style = Mock()
    window.viewport.refresh_operator_overlays = Mock()
    window.viewport.commit = Mock()
    window.line_plot_panels["surface-line"].refresh = Mock()

    window._on_control_changed(window.scene.controls["slice_position"], 0.4)

    window.viewport.refresh_surface_visual.assert_not_called()
    window.viewport.refresh_surface_axes_geometry.assert_not_called()
    window.viewport.refresh_surface_axes_style.assert_not_called()
    window.viewport.refresh_operator_overlays.assert_called_once()
    window.viewport.commit.assert_called_once()
    window.line_plot_panels["surface-line"].refresh.assert_called_once()

    window.close()
    app.quit()


def test_surface_axis_style_control_change_avoids_geometry_refresh():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = VispyFrontendWindow(build_surface_axes_binding_app())
    window.timer.stop()
    window.viewport.refresh_surface_visual = Mock()
    window.viewport.refresh_surface_axes_geometry = Mock()
    window.viewport.refresh_surface_axes_style = Mock()
    window.viewport.refresh_operator_overlays = Mock()
    window.viewport.commit = Mock()

    window._on_control_changed(window.scene.controls["tick_label_size"], 18.0)

    window.viewport.refresh_surface_visual.assert_not_called()
    window.viewport.refresh_surface_axes_geometry.assert_not_called()
    window.viewport.refresh_surface_axes_style.assert_called_once()
    window.viewport.refresh_operator_overlays.assert_not_called()
    window.viewport.commit.assert_called_once()

    window.close()
    app.quit()


def test_surface_slice_overlay_moves_with_control_changes():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = VispyFrontendWindow(build_surface_cross_section_app())
    window.timer.stop()

    overlay = window.viewport.renderer_surface._slice_overlays["surface-slice"]
    initial_pos = np.array(overlay._line._pos, copy=True)
    window._on_control_changed(window.scene.controls["slice_position"], 1.0)
    overlay = window.viewport.renderer_surface._slice_overlays["surface-slice"]
    moved_pos = np.array(overlay._line._pos, copy=True)
    window._on_control_changed(window.scene.controls["slice_axis"], "y")
    overlay = window.viewport.renderer_surface._slice_overlays["surface-slice"]
    axis_switched_pos = np.array(overlay._line._pos, copy=True)

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
        "surface-view:color_map": "bwr",
        "surface-view:color_limits": None,
        "surface-view:color_by": "height",
        "surface-view:surface_color": (0.5, 0.6, 0.8, 1.0),
        "surface-view:surface_shading": "unlit",
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


def test_surface_axes_overlay_reuses_visual_objects_for_style_refresh():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = Viewport3DPanel()
    x = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    y = np.linspace(-2.0, 2.0, 6, dtype=np.float32)
    values = (np.sin(x[None, :]) + np.cos(y[:, None])).astype(np.float32)
    field, geometry = grid_field(field_id="surface", values=values, x_coords=x, y_coords=y)
    view = SurfaceViewSpec(
        id="surface-view",
        field_id=field.id,
        geometry_id=geometry.id,
        render_axes=True,
        axes_in_middle=True,
        tick_count=7,
        tick_label_size=12.0,
        axis_label_size=16.0,
        axis_color="black",
        text_color="black",
        axis_alpha=1.0,
        axis_labels=("x", "y", "height"),
    )
    state = {
        "surface-view:color_map": "bwr",
        "surface-view:color_limits": None,
        "surface-view:color_by": "height",
        "surface-view:surface_color": (0.5, 0.6, 0.8, 1.0),
        "surface-view:surface_shading": "unlit",
        "surface-view:surface_alpha": 1.0,
        "surface-view:background_color": "white",
        "surface-view:render_axes": True,
        "surface-view:axes_in_middle": True,
        "surface-view:tick_count": 7,
        "surface-view:tick_length_scale": 1.0,
        "surface-view:tick_label_size": 12.0,
        "surface-view:axis_label_size": 16.0,
        "surface-view:axis_color": "black",
        "surface-view:text_color": "black",
        "surface-view:axis_alpha": 1.0,
    }

    panel.refresh_surface_visual(surface_view=view, surface_field=field, grid_geometry=geometry, resolved_state=state)
    panel.refresh_surface_axes_geometry(surface_view=view, resolved_state=state)
    overlay = panel.renderer_surface.axes
    visuals = {
        "axis_lines": overlay._axis_lines,
        "tick_lines": overlay._tick_lines,
        "x_ticks": overlay._tick_labels["x"],
        "y_ticks": overlay._tick_labels["y"],
        "z_ticks": overlay._tick_labels["z"],
        "x_label": overlay._axis_labels["x"],
        "y_label": overlay._axis_labels["y"],
        "z_label": overlay._axis_labels["z"],
    }

    style_state = dict(state)
    style_state["surface-view:tick_label_size"] = 18.0
    style_state["surface-view:axis_label_size"] = 22.0
    style_state["surface-view:axis_color"] = "red"
    style_state["surface-view:text_color"] = "blue"
    style_state["surface-view:axis_alpha"] = 0.5
    panel.refresh_surface_axes_style(surface_view=view, resolved_state=style_state)

    assert overlay._axis_lines is visuals["axis_lines"]
    assert overlay._tick_lines is visuals["tick_lines"]
    assert overlay._tick_labels["x"] is visuals["x_ticks"]
    assert overlay._tick_labels["y"] is visuals["y_ticks"]
    assert overlay._tick_labels["z"] is visuals["z_ticks"]
    assert overlay._axis_labels["x"] is visuals["x_label"]
    assert overlay._axis_labels["y"] is visuals["y_label"]
    assert overlay._axis_labels["z"] is visuals["z_label"]
    assert overlay._tick_labels["x"].font_size == 18.0
    assert overlay._axis_labels["x"].font_size == 22.0

    app.quit()


def test_viewport_3d_panel_applies_host_camera_settings():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    host = View3DHostSpec(
        id="surface-host",
        view_ids=("surface-view",),
        camera_distance=90.0,
        camera_elevation=20.0,
        camera_azimuth=15.0,
    )

    panel = Viewport3DPanel(host_spec=host)

    assert panel.view.camera.distance == 90.0
    assert panel.view.camera.elevation == 20.0
    assert panel.view.camera.azimuth == 15.0
    app.quit()


def test_build_surface_app_accepts_custom_3d_hosts():
    x = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, 6, dtype=np.float32)
    values = (np.sin(x[None, :]) + np.cos(y[:, None])).astype(np.float32)
    field, geometry = grid_field(field_id="surface", values=values, x_coords=x, y_coords=y)
    surface_view = SurfaceViewSpec(id="surface-view", field_id=field.id, geometry_id=geometry.id)
    host = View3DHostSpec(
        id="surface-host",
        view_ids=("surface-view",),
        camera_distance=85.0,
        camera_elevation=25.0,
        camera_azimuth=5.0,
    )

    app_spec = build_surface_app(
        field=field,
        geometry=geometry,
        surface_view=surface_view,
        view_3d_hosts=(host,),
    )

    assert app_spec.scene is not None
    assert app_spec.scene.layout.view_3d_hosts == (host,)


def test_build_surface_app_accepts_multiple_line_views():
    x = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, 6, dtype=np.float32)
    values = (np.sin(x[None, :]) + np.cos(y[:, None])).astype(np.float32)
    field, geometry = grid_field(field_id="surface", values=values, x_coords=x, y_coords=y)
    surface_view = SurfaceViewSpec(id="surface-view", field_id=field.id, geometry_id=geometry.id)
    line_views = (
        LinePlotViewSpec(id="surface-line-x", field_id=field.id, x_dim="x", selectors={"y": 0}),
        LinePlotViewSpec(id="surface-line-y", field_id=field.id, x_dim="y", selectors={"x": 0}),
    )

    app_spec = build_surface_app(
        field=field,
        geometry=geometry,
        surface_view=surface_view,
        line_views=line_views,
    )

    assert app_spec.scene is not None
    assert app_spec.scene.layout.line_plot_view_ids == ("surface-line-x", "surface-line-y")


def test_morphology_renderer_uses_fixed_clim_without_dynamic_recaling():
    renderer = MorphologyRenderer.__new__(MorphologyRenderer)
    renderer.collection = Mock()
    renderer._color_buf = np.zeros((2, 4), dtype=np.float32)

    renderer.update_colors(np.array([5.0, 6.0], dtype=np.float32), "scalar", color_limits=(0.0, 10.0))

    colors = renderer.collection.set_colors.call_args.args[0]
    assert np.allclose(colors[:, 0], np.array([0.5, 0.6], dtype=np.float32))
    assert np.allclose(colors[:, 2], np.array([0.5, 0.4], dtype=np.float32))


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


def test_frontend_shows_loading_state_before_first_document():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = VispyFrontendWindow(AppSpec(title="Loading test"))
    window.timer.stop()

    assert window._stack.currentWidget() is window._loading_label
    assert window._loading_label.text() == "Loading visualization..."

    window.close()
    app.quit()


def test_frontend_uses_session_startup_scene_before_worker_ready(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    class BootstrapSession(BufferedSession):
        @classmethod
        def startup_scene(cls):
            field = Field(
                id="bootstrap",
                values=np.array([[0.0]], dtype=np.float32),
                dims=("series", "time"),
                coords={
                    "series": np.array(["demo"]),
                    "time": np.array([0.0], dtype=np.float32),
                },
            )
            view = LinePlotViewSpec(
                id="bootstrap-view",
                field_id="bootstrap",
                x_dim="time",
                series_dim="series",
                title="Bootstrap",
                x_label="Time",
                y_label="Value",
            )
            return Scene(
                fields={"bootstrap": field},
                geometries={},
                views={"bootstrap-view": view},
                layout=LayoutSpec(title="Bootstrap", line_plot_view_ids=("bootstrap-view",)),
            )

        def initialize(self):
            raise AssertionError("worker initialization should not be used to build the startup scene")

        def advance(self) -> None:
            return None

        def handle(self, command) -> None:
            return None

    class FakeTransport:
        def __init__(self, session, parent=None):
            self.session = session
            self.parent = parent
            self._dead = False

        def start(self):
            return None

        def stop(self):
            return None

        def poll_updates(self):
            return []

    monkeypatch.setattr(frontend_module, "PipeTransport", FakeTransport)

    window = VispyFrontendWindow(AppSpec(session=BootstrapSession, title="Bootstrap test"))
    window.timer.stop()

    assert isinstance(window.transport, FakeTransport)
    assert window.transport.session is BootstrapSession
    assert window.scene is not None
    assert window.scene.layout.line_plot_view_ids == ("bootstrap-view",)
    assert window._stack.currentWidget() is window._layout_splitter

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
    document = Scene(
        fields={"voltage": field},
        geometries={},
        views={"trace": view},
        layout=LayoutSpec(title="State patch test", line_plot_view_ids=("trace",)),
    )

    class FakeTransport:
        _dead = False

        def poll_updates(self):
            updates = [StatePatch({"selected_trace_entity_ids": ["seg-b"]})]
            self.poll_updates = lambda: []
            return updates

        def stop(self):
            return None

    window = VispyFrontendWindow(AppSpec(scene=document, title="State patch test"))
    window.timer.stop()
    window.transport = FakeTransport()

    window._poll_transport()

    assert window.state["selected_trace_entity_ids"] == ["seg-b"]
    assert "seg-b" in window.line_plot_panels["trace"]._series_items
    window.close()
    app.quit()


def test_frontend_can_render_multiple_line_plot_panels():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    field = Field(
        id="signals",
        values=np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32),
        dims=("series", "time"),
        coords={
            "series": np.array(["a", "b"]),
            "time": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        },
    )
    views = {
        "trace-a": LinePlotViewSpec(
            id="trace-a",
            field_id="signals",
            x_dim="time",
            selectors={"series": "a"},
            title="Trace A",
        ),
        "trace-b": LinePlotViewSpec(
            id="trace-b",
            field_id="signals",
            x_dim="time",
            selectors={"series": "b"},
            title="Trace B",
        ),
    }
    scene = Scene(
        fields={"signals": field},
        geometries={},
        views=views,
        layout=LayoutSpec(title="Multi plot", line_plot_view_ids=("trace-a", "trace-b")),
    )

    window = VispyFrontendWindow(AppSpec(scene=scene, title="Multi plot"))
    window.timer.stop()

    assert tuple(window.line_plot_panels.keys()) == ("trace-a", "trace-b")
    x_a, y_a = window.line_plot_panel("trace-a")._plot_item.getData()
    x_b, y_b = window.line_plot_panel("trace-b")._plot_item.getData()
    assert np.allclose(x_a, np.array([0.0, 1.0, 2.0], dtype=np.float32))
    assert np.allclose(y_a, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert np.allclose(x_b, np.array([0.0, 1.0, 2.0], dtype=np.float32))
    assert np.allclose(y_b, np.array([10.0, 20.0, 30.0], dtype=np.float32))

    window.close()
    app.quit()


def test_refresh_planner_targets_only_matching_line_plot_view_for_field_replace():
    scene = Scene(
        fields={
            "field-a": Field(
                id="field-a",
                values=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                dims=("time",),
                coords={"time": np.array([0.0, 1.0, 2.0], dtype=np.float32)},
            ),
            "field-b": Field(
                id="field-b",
                values=np.array([10.0, 20.0, 30.0], dtype=np.float32),
                dims=("time",),
                coords={"time": np.array([0.0, 1.0, 2.0], dtype=np.float32)},
            ),
        },
        geometries={},
        views={
            "trace-a": LinePlotViewSpec(id="trace-a", field_id="field-a", x_dim="time"),
            "trace-b": LinePlotViewSpec(id="trace-b", field_id="field-b", x_dim="time"),
        },
        layout=LayoutSpec(title="Multi planner", line_plot_view_ids=("trace-a", "trace-b")),
    )
    planner = RefreshPlanner(scene)

    assert planner.targets_for_field_replace("field-a") == {RefreshTarget.line_plot("trace-a")}
    assert planner.targets_for_field_replace("field-b") == {RefreshTarget.line_plot("trace-b")}


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
        id="segment_history",
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


def test_multi_series_rolling_window_uses_visible_finite_history_for_x_range():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = LinePlotPanel()
    field = Field(
        id="segment_history",
        values=np.array([[np.nan, np.nan, 30.0, 40.0]], dtype=np.float32),
        dims=("segment", "time"),
        coords={
            "segment": np.array(["seg-b"]),
            "time": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        },
    )
    view = LinePlotViewSpec(
        id="trace",
        field_id=field.id,
        x_dim="time",
        series_dim="segment",
        selectors={"segment": StateBinding("selected_trace_entity_ids")},
        rolling_window=10.0,
    )

    panel.refresh(view, field, {"selected_trace_entity_ids": ["seg-b"]}, {})
    view_range = panel.plotItem.getViewBox().viewRange()

    assert np.allclose(view_range[0], [2.0, 3.0])
    app.quit()


def test_line_plot_panel_ignores_missing_selected_trace_labels_for_sparse_history():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = LinePlotPanel()
    field = Field(
        id="segment_history",
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


def test_line_plot_panel_updates_manual_ticks_when_window_crosses_minor_spacing():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = LinePlotPanel()
    view = LinePlotViewSpec(
        id="trace-plot",
        field_id="trace",
        x_dim="time",
        x_major_tick_spacing=5.0,
        x_minor_tick_spacing=1.0,
        rolling_window=2.9,
    )
    field_before = Field(
        id="trace",
        values=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        dims=("time",),
        coords={"time": np.array([0.1, 1.1, 3.0], dtype=np.float32)},
    )
    field_after = Field(
        id="trace",
        values=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        dims=("time",),
        coords={"time": np.array([1.1, 2.1, 4.0], dtype=np.float32)},
    )

    panel.refresh(view, field_before, {}, {})
    panel.refresh(view, field_after, {}, {})
    ticks = panel.plotItem.getAxis("bottom")._tickLevels

    assert ticks is not None
    minor = ticks[1]
    assert [value for value, _ in minor] == [2.0, 3.0, 4.0]
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
    document = Scene(
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
        layout=LayoutSpec(title="Cascade", main_3d_view_id=None, line_plot_view_ids=("cascade-plot",)),
    )
    window = VispyFrontendWindow(AppSpec(scene=document, title="Cascade"))
    window.timer.stop()

    assert window.viewport is None
    assert window.controls_host.isHidden()
    assert isinstance(window.centralWidget(), QtWidgets.QStackedWidget)
    assert window._stack.currentWidget() is window._layout_splitter
    assert window._layout_splitter.orientation() == QtCore.Qt.Orientation.Vertical
    assert "cascade-plot" in window.line_plot_panels

    window.close()
    app.quit()


def test_frontend_uses_splitters_for_draggable_panel_resize():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = VispyFrontendWindow(build_surface_cross_section_app())
    window.timer.stop()

    assert isinstance(window.centralWidget(), QtWidgets.QStackedWidget)
    assert window._stack.currentWidget() is window._layout_splitter
    assert window._layout_splitter.orientation() == QtCore.Qt.Orientation.Vertical
    row = window._layout_splitter.widget(0)
    assert isinstance(row, QtWidgets.QSplitter)
    assert row.orientation() == QtCore.Qt.Orientation.Horizontal
    assert row.widget(0) is window.view_hosts["surface-host"]
    assert isinstance(row.widget(1), LinePlotHostPanel)
    assert row.widget(1).line_plot_panel is window.line_plot_panels["surface-line"]
    assert window._layout_splitter.widget(1) is window.controls_host
    assert window.controls_host.controls_panel is window.controls_panel
    assert not hasattr(window, "controls")
    assert not hasattr(window, "control_host")
    assert not hasattr(window, "line_plot")
    assert not hasattr(window, "line_plots")
    assert not hasattr(window, "line_plot_for")
    assert not hasattr(window.controls_host, "controls")
    assert not window._layout_splitter.opaqueResize()
    assert not row.opaqueResize()

    window.close()
    app.quit()


def test_frontend_supports_multiple_3d_viewports():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    morph_geometry = MorphologyGeometry(
        id="morphology-geometry",
        positions=np.zeros((1, 3), dtype=np.float32),
        orientations=np.eye(3, dtype=np.float32)[None, :, :],
        radii=np.ones(1, dtype=np.float32),
        lengths=np.ones(1, dtype=np.float32),
        entity_ids=("seg-a",),
        section_names=("sec-a",),
        xlocs=np.array([0.5], dtype=np.float32),
        labels=("sec-a@0.5",),
    )
    morph_field = Field(
        id="voltage",
        values=np.array([1.0], dtype=np.float32),
        dims=("segment",),
        coords={"segment": np.array(["seg-a"])},
    )
    surface_field, surface_geometry = grid_field(
        field_id="surface",
        values=np.ones((4, 4), dtype=np.float32),
        x_coords=np.linspace(-1.0, 1.0, 4, dtype=np.float32),
        y_coords=np.linspace(-1.0, 1.0, 4, dtype=np.float32),
    )
    document = Scene(
        fields={morph_field.id: morph_field, surface_field.id: surface_field},
        geometries={morph_geometry.id: morph_geometry, surface_geometry.id: surface_geometry},
        views={
            "morphology": MorphologyViewSpec(id="morphology", geometry_id=morph_geometry.id, color_field_id=morph_field.id),
            "surface-view": SurfaceViewSpec(id="surface-view", field_id=surface_field.id, geometry_id=surface_geometry.id),
        },
        layout=LayoutSpec(title="Multi 3D", view_3d_ids=("morphology", "surface-view")),
    )
    window = VispyFrontendWindow(AppSpec(scene=document, title="Multi 3D"))
    window.timer.stop()

    assert set(window.viewports.keys()) == {"morphology", "surface-view"}
    assert window.viewport_for("morphology") is not None
    assert window.viewport_for("surface-view") is not None
    row = window._layout_splitter.widget(0)
    assert isinstance(row, QtWidgets.QSplitter)
    assert row.count() == 2
    assert window.controls_host.isHidden()

    window.close()
    app.quit()


def test_layout_spec_derives_default_3d_hosts_from_view_ids():
    layout = LayoutSpec(title="Derived hosts", view_3d_ids=("morphology", "surface"))
    document = Scene(fields={}, geometries={}, views={}, layout=layout)

    assert document.layout.view_3d_ids == ("morphology", "surface")
    assert tuple(host.id for host in document.layout.view_3d_hosts) == ("morphology", "surface")
    assert tuple(host.view_ids for host in document.layout.view_3d_hosts) == (("morphology",), ("surface",))


def test_frontend_builds_explicit_independent_3d_hosts():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    geometry = MorphologyGeometry(
        id="morphology-geometry",
        positions=np.zeros((1, 3), dtype=np.float32),
        orientations=np.eye(3, dtype=np.float32)[None, :, :],
        radii=np.ones(1, dtype=np.float32),
        lengths=np.ones(1, dtype=np.float32),
        entity_ids=("seg-a",),
        section_names=("sec-a",),
        xlocs=np.array([0.5], dtype=np.float32),
        labels=("sec-a@0.5",),
    )
    field = Field(
        id="display",
        values=np.array([1.0], dtype=np.float32),
        dims=("segment",),
        coords={"segment": np.array(["seg-a"])},
    )
    document = Scene(
        fields={field.id: field},
        geometries={geometry.id: geometry},
        views={"morphology": MorphologyViewSpec(id="morphology", geometry_id=geometry.id, color_field_id=field.id)},
        layout=LayoutSpec(
            title="Hosted",
            view_3d_hosts=(View3DHostSpec(id="main-host", view_ids=("morphology",), title="Primary 3D"),),
        ),
    )

    window = VispyFrontendWindow(AppSpec(scene=document, title="Hosted"))
    window.timer.stop()

    assert set(window.view_hosts.keys()) == {"main-host"}
    assert window._view_to_host_id["morphology"] == "main-host"
    assert window.view_hosts["main-host"].title() == "Primary 3D"
    assert window.viewport_for("morphology") is window.view_hosts["main-host"].viewport

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
    document = Scene(
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
    window = VispyFrontendWindow(AppSpec(scene=document, title="Action test"))
    window.timer.stop()
    window.transport = Mock()
    window.state["selected_entity_id"] = "seg-a"

    action_button = window.controls_panel.widgets["mark_selected"]
    action_button.click()

    window.transport.send_command.assert_called_once()
    command = window.transport.send_command.call_args[0][0]
    assert command.action_id == "mark_selected"
    assert command.payload["entity_id"] == "seg-a"

    window.close()
    app.quit()


def test_controls_host_title_reflects_controls_and_actions_presence():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = VispyFrontendWindow(build_surface_cross_section_app())
    window.timer.stop()

    assert window.controls_host.title() == "Controls"

    actions_only_document = Scene(
        fields={},
        geometries={},
        views={},
        actions={"reset": ActionSpec(id="reset", label="Reset")},
        layout=LayoutSpec(title="Actions only", action_ids=("reset",)),
    )
    window._set_scene(actions_only_document)
    assert window.controls_host.title() == "Actions"

    controls_and_actions_document = Scene(
        fields={},
        geometries={},
        views={},
        controls={"gain": ControlSpec(id="gain", kind="float", label="Gain", default=1.0)},
        actions={"reset": ActionSpec(id="reset", label="Reset")},
        layout=LayoutSpec(title="Controls and actions", control_ids=("gain",), action_ids=("reset",)),
    )
    window._set_scene(controls_and_actions_document)
    assert window.controls_host.title() == "Controls & Actions"

    window.close()
    app.quit()


def test_action_shortcut_dispatches_invoke_action():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    document = Scene(
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
    window = VispyFrontendWindow(AppSpec(scene=document, title="Shortcut test"))
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
    document = Scene(
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
    window = VispyFrontendWindow(AppSpec(scene=document, title="Selection action test"))
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
    document = Scene(
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
    window = VispyFrontendWindow(AppSpec(scene=document, interaction_target=interaction_target, title="Callback interaction test"))
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
    document = Scene(
        fields={field.id: field},
        geometries={geometry.id: geometry},
        views={"morphology": MorphologyViewSpec(id="morphology", geometry_id=geometry.id)},
        layout=LayoutSpec(title="Entity info test", main_3d_view_id="morphology"),
    )
    window = VispyFrontendWindow(AppSpec(scene=document, title="Entity info test"))
    window.timer.stop()

    info = window._interaction_context().entity_info("seg-b")

    assert info is not None
    assert info["entity_id"] == "seg-b"
    assert info["section_name"] == "sec-b"
    assert info["label"] == "sec-b@0.9"

    window.close()
    app.quit()


def test_frontend_reset_action_sends_reset_command():
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
    document = Scene(
        fields={field.id: field},
        geometries={geometry.id: geometry},
        views={"morphology": MorphologyViewSpec(id="morphology", geometry_id=geometry.id)},
        actions={"reset": ActionSpec(id="reset", label="Reset")},
        layout=LayoutSpec(title="Reset button test", main_3d_view_id="morphology", action_ids=("reset",)),
    )
    window = VispyFrontendWindow(AppSpec(scene=document, title="Reset button test"))
    window.timer.stop()
    window.transport = Mock()

    window._on_action_invoked(document.actions["reset"], {})

    window.transport.send_command.assert_called_once()
    command = window.transport.send_command.call_args[0][0]
    assert isinstance(command, Reset)

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

    document = session.build_scene(
        geometry=geometry,
        display_values=np.array([1.0], dtype=np.float32),
        time_value=0.0,
    )

    assert document.layout.control_ids == ("gain",)
    assert document.layout.action_ids == ("toggle_mode",)
    trace_view = document.views["trace"]
    assert trace_view.rolling_window == 25.0
    assert trace_view.trim_to_rolling_window is True


def test_neuron_session_defaults_to_batched_display_updates():
    session = DummyNeuronSession()

    assert session.steps_per_update() == 1


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
    document = session.build_scene(
        geometry=geometry,
        display_values=np.array([1.0], dtype=np.float32),
        time_value=0.0,
    )

    assert session._resolved_field_max_samples(document, field_id=NeuronSceneBuilder.HISTORY_FIELD_ID, append_dim="time") == 1201


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
    document = Scene(
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
        layout=LayoutSpec(title="Append test", main_3d_view_id="morphology", line_plot_view_ids=("trace",)),
    )
    window = VispyFrontendWindow(AppSpec(scene=document, title="Append test"))
    window.timer.stop()
    window.state["selected_entity_id"] = "seg-a"
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

    updated = window.scene.fields["voltage"]
    assert updated.coord("time").tolist() == [1.0, 2.0]
    assert np.allclose(updated.values, np.array([[2.0, 3.0], [20.0, 30.0]], dtype=np.float32))
    x_data, y_data = window.line_plot_panels["trace"]._plot_item.getData()
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
    document = Scene(
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
        layout=LayoutSpec(title="Batch append test", main_3d_view_id="morphology", line_plot_view_ids=("trace",)),
    )
    window = VispyFrontendWindow(AppSpec(scene=document, title="Batch append test"))
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

    with patch.object(Field, "append", autospec=True, side_effect=Field.append) as append_spy:
        window._poll_transport()

    updated = window.scene.fields["voltage"]
    assert updated.coord("time").tolist() == [0.0, 1.0, 2.0]
    assert np.allclose(updated.values, np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32))
    append_spy.assert_called_once()
    window._refresh_morphology.assert_called_once()
    window._refresh_line_plot.assert_called_once()
    window.viewport.commit.assert_called_once()

    window.close()
    app.quit()


def test_frontend_flushes_buffered_field_appends_before_field_replace():
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
    document = Scene(
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
        layout=LayoutSpec(title="Append replace order test", main_3d_view_id="morphology", line_plot_view_ids=("trace",)),
    )
    window = VispyFrontendWindow(AppSpec(scene=document, title="Append replace order test"))
    window.timer.stop()
    window.state["selected_entity_id"] = "seg-a"
    window.transport = Mock()
    window.transport.poll_updates.return_value = [
        FieldAppend(
            field_id="voltage",
            append_dim="time",
            values=np.array([[2.0], [20.0]], dtype=np.float32),
            coord_values=np.array([1.0], dtype=np.float32),
            max_length=4,
        ),
        FieldReplace(
            field_id="voltage",
            values=np.array([[100.0], [200.0]], dtype=np.float32),
            coords={
                "segment": np.array(["seg-a", "seg-b"]),
                "time": np.array([5.0], dtype=np.float32),
            },
        ),
        FieldAppend(
            field_id="voltage",
            append_dim="time",
            values=np.array([[300.0], [400.0]], dtype=np.float32),
            coord_values=np.array([6.0], dtype=np.float32),
            max_length=4,
        ),
    ]

    window._poll_transport()

    updated = window.scene.fields["voltage"]
    assert updated.coord("time").tolist() == [5.0, 6.0]
    assert np.allclose(updated.values, np.array([[100.0, 300.0], [200.0, 400.0]], dtype=np.float32))

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
