from __future__ import annotations

import sys
from enum import Enum, auto
from typing import Any

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from vispy import app as vispy_app, use

use(app="pyqt6", gl="gl+")

from compneurovis.core import AppSpec, Document, MorphologyGeometry, MorphologyViewSpec, StateBinding, SurfaceViewSpec
from compneurovis.frontends.vispy.panels import ControlsPanel, LinePlotPanel, Viewport3DPanel
from compneurovis.session import DocumentPatch, DocumentReady, FieldUpdate, PipeTransport, Reset, SetControl, configure_multiprocessing


class RefreshTarget(Enum):
    CONTROLS = auto()
    MORPHOLOGY = auto()
    SURFACE_VISUAL = auto()
    SURFACE_AXES = auto()
    SURFACE_SLICE = auto()
    LINE_PLOT = auto()


class RefreshPlanner:
    SURFACE_VISUAL_PROPS = frozenset({"field_id", "geometry_id", "cmap", "clim", "color_by", "surface_alpha", "background_color"})
    SURFACE_AXES_PROPS = frozenset(
        {
            "field_id",
            "geometry_id",
            "render_axes",
            "axes_in_middle",
            "tick_count",
            "tick_length_scale",
            "tick_label_size",
            "axis_label_size",
            "axis_color",
            "text_color",
            "axis_alpha",
            "axis_labels",
        }
    )
    SURFACE_SLICE_PROPS = frozenset({"field_id", "geometry_id", "slice_axis_state_key", "slice_position_state_key", "slice_color", "slice_alpha", "slice_width"})
    LINE_PLOT_PROPS = frozenset(
        {
            "field_id",
            "x_dim",
            "selectors",
            "orthogonal_slice_state_key",
            "orthogonal_position_state_key",
            "x_label",
            "y_label",
            "x_unit",
            "y_unit",
            "pen",
            "background_color",
            "title",
        }
    )

    def __init__(self, document: Document):
        self.document = document

    def _main_view(self):
        if self.document.layout.main_3d_view_id is None:
            return None
        return self.document.views.get(self.document.layout.main_3d_view_id)

    def morphology_view(self):
        view = self._main_view()
        return view if isinstance(view, MorphologyViewSpec) else None

    def surface_view(self):
        view = self._main_view()
        return view if isinstance(view, SurfaceViewSpec) else None

    def line_view(self):
        if self.document.layout.line_plot_view_id is None:
            return None
        return self.document.views.get(self.document.layout.line_plot_view_id)

    def full_refresh_targets(self) -> set[RefreshTarget]:
        targets = {RefreshTarget.CONTROLS}
        if self.morphology_view() is not None:
            targets.add(RefreshTarget.MORPHOLOGY)
        if self.surface_view() is not None:
            targets.update({RefreshTarget.SURFACE_VISUAL, RefreshTarget.SURFACE_AXES, RefreshTarget.SURFACE_SLICE})
        if self.line_view() is not None:
            targets.add(RefreshTarget.LINE_PLOT)
        return targets

    def targets_for_state_change(self, state_key: str) -> set[RefreshTarget]:
        targets: set[RefreshTarget] = set()

        morph_view = self.morphology_view()
        if morph_view is not None and binding_key(morph_view.background_color) == state_key:
            targets.add(RefreshTarget.MORPHOLOGY)

        surface_view = self.surface_view()
        if surface_view is not None:
            if any(binding_key(getattr(surface_view, prop)) == state_key for prop in self.SURFACE_VISUAL_PROPS if hasattr(surface_view, prop)):
                targets.add(RefreshTarget.SURFACE_VISUAL)
            if any(binding_key(getattr(surface_view, prop)) == state_key for prop in self.SURFACE_AXES_PROPS if hasattr(surface_view, prop)):
                targets.add(RefreshTarget.SURFACE_AXES)
            if any(binding_key(getattr(surface_view, prop)) == state_key for prop in self.SURFACE_SLICE_PROPS if hasattr(surface_view, prop)):
                targets.add(RefreshTarget.SURFACE_SLICE)
            if state_key in {surface_view.slice_axis_state_key, surface_view.slice_position_state_key}:
                targets.add(RefreshTarget.SURFACE_SLICE)

        line_view = self.line_view()
        if isinstance(line_view, MorphologyViewSpec):
            return targets
        if line_view is not None:
            if state_key in {line_view.orthogonal_slice_state_key, line_view.orthogonal_position_state_key}:
                targets.add(RefreshTarget.LINE_PLOT)
            if any(binding_key(value) == state_key for value in line_view.selectors.values()):
                targets.add(RefreshTarget.LINE_PLOT)
            if any(binding_key(getattr(line_view, prop)) == state_key for prop in ("pen", "background_color")):
                targets.add(RefreshTarget.LINE_PLOT)

        return targets

    def targets_for_field_update(self, field_id: str) -> set[RefreshTarget]:
        targets: set[RefreshTarget] = set()
        morph_view = self.morphology_view()
        if morph_view is not None and morph_view.color_field_id == field_id:
            targets.add(RefreshTarget.MORPHOLOGY)

        surface_view = self.surface_view()
        if surface_view is not None and surface_view.field_id == field_id:
            targets.update({RefreshTarget.SURFACE_VISUAL, RefreshTarget.SURFACE_AXES, RefreshTarget.SURFACE_SLICE})

        line_view = self.line_view()
        if line_view is not None and getattr(line_view, "field_id", None) == field_id:
            targets.add(RefreshTarget.LINE_PLOT)
        return targets

    def targets_for_view_patch(self, view_id: str, changed_props: set[str]) -> set[RefreshTarget]:
        view = self.document.views.get(view_id)
        if isinstance(view, MorphologyViewSpec):
            return {RefreshTarget.MORPHOLOGY}
        if isinstance(view, SurfaceViewSpec):
            targets: set[RefreshTarget] = set()
            if changed_props & self.SURFACE_VISUAL_PROPS:
                targets.add(RefreshTarget.SURFACE_VISUAL)
            if changed_props & self.SURFACE_AXES_PROPS:
                targets.add(RefreshTarget.SURFACE_AXES)
            if changed_props & self.SURFACE_SLICE_PROPS:
                targets.add(RefreshTarget.SURFACE_SLICE)
            return targets
        if view is not None and changed_props & self.LINE_PLOT_PROPS:
            return {RefreshTarget.LINE_PLOT}
        return set()


class VispyFrontendWindow(QtWidgets.QMainWindow):
    def __init__(self, app_spec: AppSpec):
        super().__init__()
        self.app_spec = app_spec
        self.document: Document | None = None
        self.state: dict[str, Any] = {}
        self.transport: PipeTransport | None = None
        self.refresh_planner: RefreshPlanner | None = None

        self.viewport = Viewport3DPanel(on_entity_selected=self._on_entity_selected)
        self.line_plot = LinePlotPanel()
        self.controls = ControlsPanel(self._on_control_changed)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.addWidget(self.viewport, 2)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.line_plot, 3)
        right_layout.addWidget(self.controls, 2)
        layout.addWidget(right, 1)

        self.setCentralWidget(central)
        self.resize(1280, 720)
        self.statusBar().showMessage("Starting CompNeuroVis")

        if app_spec.document is not None:
            self._set_document(app_spec.document)

        if app_spec.session is not None:
            configure_multiprocessing()
            self.transport = PipeTransport(app_spec.session, parent=self)
            self.transport.start()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll_transport)
        self.timer.start(1000 // 60)

    def _set_document(self, document: Document) -> None:
        self.document = document
        self.refresh_planner = RefreshPlanner(document)
        self.setWindowTitle(self.app_spec.title or document.layout.title)
        for control in document.controls.values():
            self.state.setdefault(control.resolved_state_key(), control.default)

        morphology_view = self._morphology_view()
        if morphology_view is not None:
            geometry = document.geometries[morphology_view.geometry_id]
            if isinstance(geometry, MorphologyGeometry) and geometry.entity_ids:
                self.state.setdefault("selected_entity_id", geometry.entity_ids[0])
                self.state.setdefault("selected_entity_label", geometry.label_for(geometry.entity_ids[0]))

        self.viewport.clear()
        self._apply_refresh_targets(self.refresh_planner.full_refresh_targets())

    def _morphology_view(self):
        if self.document is None or self.document.layout.main_3d_view_id is None:
            return None
        view = self.document.views.get(self.document.layout.main_3d_view_id)
        return view if isinstance(view, MorphologyViewSpec) else None

    def _surface_view(self):
        if self.document is None or self.document.layout.main_3d_view_id is None:
            return None
        view = self.document.views.get(self.document.layout.main_3d_view_id)
        return view if isinstance(view, SurfaceViewSpec) else None

    def _line_view(self):
        if self.document is None or self.document.layout.line_plot_view_id is None:
            return None
        return self.document.views.get(self.document.layout.line_plot_view_id)

    def _refresh_controls(self) -> None:
        if self.document is None:
            return
        controls = [self.document.controls[control_id] for control_id in self.document.layout.control_ids if control_id in self.document.controls]
        self.controls.set_controls(controls, self.state)

    def _resolved_morphology_state(self, view: MorphologyViewSpec) -> dict[str, Any]:
        return {
            f"{view.id}:background_color": resolve_value(view.background_color, self.state),
        }

    def _resolved_surface_state(self, view: SurfaceViewSpec) -> dict[str, Any]:
        resolved = {
            f"{view.id}:cmap": view.cmap,
            f"{view.id}:clim": view.clim,
            f"{view.id}:color_by": view.color_by,
            f"{view.id}:surface_alpha": resolve_value(view.surface_alpha, self.state),
            f"{view.id}:background_color": resolve_value(view.background_color, self.state),
            f"{view.id}:render_axes": resolve_value(view.render_axes, self.state),
            f"{view.id}:axes_in_middle": resolve_value(view.axes_in_middle, self.state),
            f"{view.id}:tick_count": resolve_value(view.tick_count, self.state),
            f"{view.id}:tick_length_scale": resolve_value(view.tick_length_scale, self.state),
            f"{view.id}:tick_label_size": resolve_value(view.tick_label_size, self.state),
            f"{view.id}:axis_label_size": resolve_value(view.axis_label_size, self.state),
            f"{view.id}:axis_color": resolve_value(view.axis_color, self.state),
            f"{view.id}:text_color": resolve_value(view.text_color, self.state),
            f"{view.id}:axis_alpha": resolve_value(view.axis_alpha, self.state),
            f"{view.id}:slice_color": resolve_value(view.slice_color, self.state),
            f"{view.id}:slice_alpha": resolve_value(view.slice_alpha, self.state),
            f"{view.id}:slice_width": resolve_value(view.slice_width, self.state),
        }
        if view.slice_axis_state_key is not None:
            resolved[view.slice_axis_state_key] = self.state.get(view.slice_axis_state_key)
        if view.slice_position_state_key is not None:
            resolved[view.slice_position_state_key] = self.state.get(view.slice_position_state_key)
        return resolved

    def _refresh_morphology(self) -> None:
        if self.document is None:
            return
        morph_view = self._morphology_view()
        morphology_geometry = None
        morphology_colors = None
        if morph_view is not None:
            geometry = self.document.geometries.get(morph_view.geometry_id)
            if isinstance(geometry, MorphologyGeometry):
                morphology_geometry = geometry
                if morph_view.color_field_id:
                    field = self.document.fields[morph_view.color_field_id]
                    if morph_view.sample_dim and morph_view.sample_dim in field.dims:
                        latest = field.select({morph_view.sample_dim: -1})
                        morphology_colors = latest.values
                    else:
                        morphology_colors = field.values
        self.viewport.refresh_morphology(
            morphology_geometry=morphology_geometry,
            morphology_view=morph_view,
            morphology_colors=morphology_colors,
            resolved_state=self._resolved_morphology_state(morph_view) if morph_view is not None else {},
        )

    def _refresh_surface_visual(self) -> None:
        if self.document is None:
            return
        surface_view = self._surface_view()
        surface_field = None
        grid_geometry = None
        if surface_view is not None:
            surface_field = self.document.fields[surface_view.field_id]
            if surface_view.geometry_id is not None:
                grid_geometry = self.document.geometries.get(surface_view.geometry_id)
        self.viewport.refresh_surface_visual(
            surface_view=surface_view,
            surface_field=surface_field,
            grid_geometry=grid_geometry,
            resolved_state=self._resolved_surface_state(surface_view) if surface_view is not None else {},
        )

    def _refresh_surface_axes(self) -> None:
        surface_view = self._surface_view()
        self.viewport.refresh_surface_axes(
            surface_view=surface_view,
            resolved_state=self._resolved_surface_state(surface_view) if surface_view is not None else {},
        )

    def _refresh_surface_slice(self) -> None:
        surface_view = self._surface_view()
        self.viewport.refresh_surface_slice(
            surface_view=surface_view,
            resolved_state=self._resolved_surface_state(surface_view) if surface_view is not None else {},
        )

    def _refresh_line_plot(self) -> None:
        if self.document is None:
            return
        line_view = self._line_view()
        geometry_lookup = {
            key: value
            for key, value in self.document.geometries.items()
            if isinstance(value, MorphologyGeometry)
        }
        line_field = self.document.fields.get(line_view.field_id) if line_view is not None else None
        self.line_plot.refresh(line_view, line_field, self.state, geometry_lookup)

    def _apply_refresh_targets(self, targets: set[RefreshTarget]) -> None:
        if not targets:
            return

        if RefreshTarget.CONTROLS in targets:
            self._refresh_controls()

        viewport_dirty = False
        if RefreshTarget.MORPHOLOGY in targets:
            self._refresh_morphology()
            viewport_dirty = True
        if RefreshTarget.SURFACE_VISUAL in targets:
            self._refresh_surface_visual()
            viewport_dirty = True
        if RefreshTarget.SURFACE_AXES in targets:
            self._refresh_surface_axes()
            viewport_dirty = True
        if RefreshTarget.SURFACE_SLICE in targets:
            self._refresh_surface_slice()
            viewport_dirty = True
        if viewport_dirty:
            self.viewport.commit()

        if RefreshTarget.LINE_PLOT in targets:
            self._refresh_line_plot()

    def _poll_transport(self) -> None:
        if self.transport is None:
            return
        for update in self.transport.poll_updates():
            if isinstance(update, DocumentReady):
                self._set_document(update.document)
                self.statusBar().showMessage("Document ready")
            elif isinstance(update, FieldUpdate):
                if self.document is None:
                    continue
                current = self.document.fields[update.field_id]
                coords = current.coords if update.coords is None else update.coords
                self.document.fields[update.field_id] = current.with_values(update.values, coords=coords, attrs_update=update.attrs_update)
                if self.refresh_planner is not None:
                    self._apply_refresh_targets(self.refresh_planner.targets_for_field_update(update.field_id))
            elif isinstance(update, DocumentPatch):
                if self.document is None:
                    continue
                targets: set[RefreshTarget] = set()
                for view_id, patch in update.view_updates.items():
                    self.document.replace_view(view_id, patch)
                    if self.refresh_planner is not None:
                        targets.update(self.refresh_planner.targets_for_view_patch(view_id, set(patch.keys())))
                for control_id, patch in update.control_updates.items():
                    self.document.replace_control(control_id, patch)
                    targets.add(RefreshTarget.CONTROLS)
                self.document.metadata.update(update.metadata_updates)
                self._apply_refresh_targets(targets)
            else:
                message = getattr(update, "message", str(update))
                self.statusBar().showMessage(message)

    def _on_entity_selected(self, entity_id: str) -> None:
        self.state["selected_entity_id"] = entity_id
        if self.document is not None:
            for geometry in self.document.geometries.values():
                if isinstance(geometry, MorphologyGeometry) and entity_id in geometry.entity_ids:
                    self.state["selected_entity_label"] = geometry.label_for(entity_id)
                    break
        if self.refresh_planner is not None:
            self._apply_refresh_targets(self.refresh_planner.targets_for_state_change("selected_entity_id"))

    def _on_control_changed(self, control, value) -> None:
        self.state[control.resolved_state_key()] = value
        if self.transport is not None and control.send_to_session:
            self.transport.send_command(SetControl(control.id, value))
        if self.refresh_planner is not None:
            self._apply_refresh_targets(self.refresh_planner.targets_for_state_change(control.resolved_state_key()))

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and self.transport is not None:
            self.transport.send_command(Reset())
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.timer.stop()
        if self.transport is not None:
            self.transport.stop()
        super().closeEvent(event)


def resolve_value(value, state: dict[str, Any]):
    if isinstance(value, StateBinding):
        return state.get(value.key)
    return value


def binding_key(value) -> str | None:
    if isinstance(value, StateBinding):
        return value.key
    return None


def run_app(app_spec: AppSpec) -> None:
    qt_app = QtWidgets.QApplication.instance()
    owns_app = qt_app is None
    if qt_app is None:
        qt_app = QtWidgets.QApplication(sys.argv)
    window = VispyFrontendWindow(app_spec)
    window.show()
    if owns_app:
        vispy_app.run()
