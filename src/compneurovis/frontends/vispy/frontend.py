from __future__ import annotations

import multiprocessing as mp
import sys
from enum import Enum, auto
from typing import Any

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from vispy import app as vispy_app, use

use(app="pyqt6", gl="gl+")

from compneurovis.core import AppSpec, Document, MorphologyGeometry, MorphologyViewSpec, StateBinding, SurfaceViewSpec
from compneurovis.frontends.vispy.panels import ControlsPanel, LinePlotPanel, Viewport3DPanel
from compneurovis.session import DocumentPatch, DocumentReady, FieldAppend, FieldReplace, InvokeAction, PipeTransport, Reset, SetControl, configure_multiprocessing


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

    def targets_for_field_replace(self, field_id: str, coords_changed: bool = True) -> set[RefreshTarget]:
        targets: set[RefreshTarget] = set()
        morph_view = self.morphology_view()
        if morph_view is not None and morph_view.color_field_id == field_id:
            targets.add(RefreshTarget.MORPHOLOGY)

        surface_view = self.surface_view()
        if surface_view is not None and surface_view.field_id == field_id:
            targets.add(RefreshTarget.SURFACE_VISUAL)
            if coords_changed or surface_view.clim is None:
                # Axes z-ticks depend on z range — safe to skip when coords are unchanged and
                # clim is fixed (z-range for display is constant). ~48 VisPy objects rebuilt otherwise.
                targets.add(RefreshTarget.SURFACE_AXES)
            if coords_changed or surface_view.clim is None:
                targets.add(RefreshTarget.SURFACE_SLICE)

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
        self._active_selection_action_id: str | None = None
        self.interaction_target = app_spec.interaction_target if app_spec.interaction_target is not None else app_spec.session

        self.viewport = Viewport3DPanel(on_entity_selected=self._on_entity_selected)
        self.line_plot = LinePlotPanel()
        self.controls = ControlsPanel(self._on_control_changed, self._on_action_invoked)

        self._right_splitter = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        self._right_splitter.setChildrenCollapsible(False)
        self._right_splitter.setOpaqueResize(False)
        self._right_splitter.addWidget(self.line_plot)
        self._right_splitter.addWidget(self.controls)
        self._right_splitter.setStretchFactor(0, 3)
        self._right_splitter.setStretchFactor(1, 2)

        self._horizontal_splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        self._horizontal_splitter.setChildrenCollapsible(False)
        self._horizontal_splitter.setOpaqueResize(False)
        self._horizontal_splitter.addWidget(self.viewport)
        self._horizontal_splitter.addWidget(self._right_splitter)
        self._horizontal_splitter.setStretchFactor(0, 2)
        self._horizontal_splitter.setStretchFactor(1, 1)

        self.setCentralWidget(self._horizontal_splitter)
        self.resize(1280, 720)
        self.statusBar().showMessage("Starting CompNeuroVis")
        self._apply_default_splitter_sizes(has_3d=True)

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
        self._active_selection_action_id = None
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
        self._update_panel_visibility()
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

    def _update_panel_visibility(self) -> None:
        has_3d = self.document is not None and self.document.layout.main_3d_view_id is not None
        self.viewport.setVisible(has_3d)
        self._apply_default_splitter_sizes(has_3d=has_3d)

    def _apply_default_splitter_sizes(self, *, has_3d: bool) -> None:
        width = max(self.width(), 1)
        height = max(self.height(), 1)
        self._right_splitter.setSizes([max(1, int(height * 0.6)), max(1, int(height * 0.4))])
        if has_3d:
            self._horizontal_splitter.setSizes([max(1, int(width * 0.67)), max(1, int(width * 0.33))])
        else:
            self._horizontal_splitter.setSizes([0, width])

    def _refresh_controls(self) -> None:
        if self.document is None:
            return
        controls = [self.document.controls[control_id] for control_id in self.document.layout.control_ids if control_id in self.document.controls]
        actions = [self.document.actions[action_id] for action_id in self.document.layout.action_ids if action_id in self.document.actions]
        self.controls.set_controls(controls, actions, self.state)

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
        pending_targets: set[RefreshTarget] = set()
        pending_status: str | None = None
        for update in self.transport.poll_updates():
            if isinstance(update, DocumentReady):
                self._set_document(update.document)
                pending_targets.clear()
                pending_status = "Document ready"
            elif isinstance(update, FieldReplace):
                if self.document is None:
                    continue
                current = self.document.fields[update.field_id]
                coords_changed = update.coords is not None
                coords = current.coords if update.coords is None else update.coords
                self.document.fields[update.field_id] = current.with_values(update.values, coords=coords, attrs_update=update.attrs_update)
                if self.refresh_planner is not None:
                    pending_targets.update(self.refresh_planner.targets_for_field_replace(update.field_id, coords_changed=coords_changed))
            elif isinstance(update, FieldAppend):
                if self.document is None:
                    continue
                current = self.document.fields[update.field_id]
                self.document.fields[update.field_id] = current.append(
                    update.append_dim,
                    update.values,
                    update.coord_values,
                    max_length=update.max_length,
                    attrs_update=update.attrs_update,
                )
                if self.refresh_planner is not None:
                    pending_targets.update(self.refresh_planner.targets_for_field_replace(update.field_id))
            elif isinstance(update, DocumentPatch):
                if self.document is None:
                    continue
                for view_id, patch in update.view_updates.items():
                    self.document.replace_view(view_id, patch)
                    if self.refresh_planner is not None:
                        pending_targets.update(self.refresh_planner.targets_for_view_patch(view_id, set(patch.keys())))
                for control_id, patch in update.control_updates.items():
                    self.document.replace_control(control_id, patch)
                    pending_targets.add(RefreshTarget.CONTROLS)
                self.document.metadata.update(update.metadata_updates)
            else:
                pending_status = getattr(update, "message", str(update))
        if pending_targets:
            self._apply_refresh_targets(pending_targets)
        if pending_status is not None:
            self.statusBar().showMessage(pending_status)

    def _on_entity_selected(self, entity_id: str) -> None:
        self.state["selected_entity_id"] = entity_id
        if self.document is not None:
            for geometry in self.document.geometries.values():
                if isinstance(geometry, MorphologyGeometry) and entity_id in geometry.entity_ids:
                    self.state["selected_entity_label"] = geometry.label_for(entity_id)
                    break
            if self._invoke_interaction_entity_click(entity_id):
                pass
            elif self._active_selection_action_id is not None:
                action = self.document.actions.get(self._active_selection_action_id)
                if action is not None:
                    payload = {
                        key: resolve_value(value, self.state)
                        for key, value in action.payload.items()
                    }
                    payload[action.selection_payload_key] = entity_id
                    self._send_action(action, payload)
        if self.refresh_planner is not None:
            self._apply_refresh_targets(self.refresh_planner.targets_for_state_change("selected_entity_id"))

    def _on_control_changed(self, control, value) -> None:
        self.state[control.resolved_state_key()] = value
        if self.transport is not None and control.send_to_session:
            self.transport.send_command(SetControl(control.id, value))
        if self.refresh_planner is not None:
            self._apply_refresh_targets(self.refresh_planner.targets_for_state_change(control.resolved_state_key()))

    def _on_action_invoked(self, action, payload: dict[str, Any]) -> None:
        if self._invoke_interaction_action(action.id, payload):
            return
        if action.selection_mode:
            self._toggle_selection_action_mode(action)
            return
        self._send_action(action, payload)

    def _send_action(self, action, payload: dict[str, Any]) -> None:
        if self.transport is not None:
            self.transport.send_command(InvokeAction(action.id, payload))

    def keyPressEvent(self, event) -> None:
        key_text = self._event_key_text(event)
        if key_text and self._invoke_interaction_key_press(key_text):
            event.accept()
            return
        if self.document is not None:
            matched_action = self._action_for_event(event)
            if matched_action is not None:
                payload = {
                    key: resolve_value(value, self.state)
                    for key, value in matched_action.payload.items()
                }
                self._on_action_invoked(matched_action, payload)
                event.accept()
                return
        if event.key() == Qt.Key.Key_Space and self.transport is not None:
            self.transport.send_command(Reset())
        super().keyPressEvent(event)

    def _action_for_event(self, event: QtGui.QKeyEvent):
        if self.document is None:
            return None
        pressed = self._event_key_text(event)
        for action_id in self.document.layout.action_ids:
            action = self.document.actions.get(action_id)
            if action is None:
                continue
            for shortcut in action.shortcuts:
                normalized = QtGui.QKeySequence(shortcut).toString(QtGui.QKeySequence.SequenceFormat.PortableText)
                if normalized and normalized == pressed:
                    return action
        return None

    def _toggle_selection_action_mode(self, action) -> None:
        if self._active_selection_action_id == action.id:
            self._active_selection_action_id = None
            self.statusBar().showMessage(f"{action.label} mode OFF")
            return
        self._active_selection_action_id = action.id
        self.statusBar().showMessage(f"{action.label} mode ON: click a segment to apply")

    def _event_key_text(self, event: QtGui.QKeyEvent) -> str:
        return QtGui.QKeySequence(event.modifiers().value | event.key()).toString(
            QtGui.QKeySequence.SequenceFormat.PortableText
        )

    def _interaction_context(self) -> "FrontendInteractionContext":
        return FrontendInteractionContext(self)

    def _invoke_interaction_action(self, action_id: str, payload: dict[str, Any]) -> bool:
        target = self.interaction_target
        handler = getattr(target, "on_action", None)
        if handler is None:
            return False
        return bool(handler(action_id, payload, self._interaction_context()))

    def _invoke_interaction_key_press(self, key: str) -> bool:
        target = self.interaction_target
        handler = getattr(target, "on_key_press", None)
        if handler is None:
            return False
        return bool(handler(key, self._interaction_context()))

    def _invoke_interaction_entity_click(self, entity_id: str) -> bool:
        target = self.interaction_target
        handler = getattr(target, "on_entity_clicked", None)
        if handler is None:
            return False
        return bool(handler(entity_id, self._interaction_context()))

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


class FrontendInteractionContext:
    def __init__(self, window: VispyFrontendWindow):
        self.window = window

    @property
    def document(self) -> Document | None:
        return self.window.document

    @property
    def selected_entity_id(self) -> str | None:
        value = self.window.state.get("selected_entity_id")
        return str(value) if value is not None else None

    def state(self, key: str, default: Any = None) -> Any:
        return self.window.state.get(key, default)

    def entity_info(self, entity_id: str | None = None) -> dict[str, Any] | None:
        current_id = entity_id or self.selected_entity_id
        if current_id is None or self.window.document is None:
            return None
        for geometry in self.window.document.geometries.values():
            if not isinstance(geometry, MorphologyGeometry):
                continue
            try:
                return geometry.entity_info(current_id)
            except KeyError:
                continue
        return None

    def set_state(self, key: str, value: Any) -> None:
        self.window.state[key] = value
        if self.window.refresh_planner is not None:
            self.window._apply_refresh_targets(self.window.refresh_planner.targets_for_state_change(key))

    def show_status(self, message: str, timeout_ms: int | None = None) -> None:
        self.window.statusBar().showMessage(message)
        if timeout_ms is not None:
            QtCore.QTimer.singleShot(timeout_ms, self.window.statusBar().clearMessage)

    def clear_status(self) -> None:
        self.window.statusBar().clearMessage()

    def invoke_action(self, action_id: str, payload: dict[str, Any] | None = None) -> None:
        if self.window.document is None:
            return
        action = self.window.document.actions.get(action_id)
        if action is None:
            return
        resolved_payload = payload if payload is not None else {
            key: resolve_value(value, self.window.state)
            for key, value in action.payload.items()
        }
        self.window._send_action(action, resolved_payload)

    def set_control(self, control_id: str, value: Any) -> None:
        if self.window.document is None:
            return
        control = self.window.document.controls.get(control_id)
        if control is None:
            return
        self.window._on_control_changed(control, value)


def run_app(app_spec: AppSpec) -> None:
    if mp.current_process().name != "MainProcess":
        return
    qt_app = QtWidgets.QApplication.instance()
    owns_app = qt_app is None
    if qt_app is None:
        qt_app = QtWidgets.QApplication(sys.argv)
    window = VispyFrontendWindow(app_spec)
    window.show()
    if owns_app:
        vispy_app.run()
