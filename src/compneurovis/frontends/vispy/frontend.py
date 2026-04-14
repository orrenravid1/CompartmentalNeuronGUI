from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import sys
from typing import Any

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from vispy import app as vispy_app, use

use(app="pyqt6", gl="gl+")

from compneurovis.core import ActionSpec, AppSpec, ControlSpec, GridSliceOperatorSpec, LinePlotViewSpec, MorphologyGeometry, MorphologyViewSpec, Scene, StateBinding, SurfaceViewSpec, View3DHostSpec
from compneurovis.frontends.vispy.panels import ControlsHostPanel, ControlsPanel, IndependentCanvas3DHostPanel, LinePlotHostPanel, LinePlotPanel, Viewport3DPanel
from compneurovis.session import ScenePatch, SceneReady, EntityClicked, FieldAppend, FieldReplace, InvokeAction, KeyPressed, PipeTransport, Reset, SetControl, StatePatch, Status, configure_multiprocessing, resolve_interaction_target_source
from compneurovis.session.base import resolve_startup_scene_source


@dataclass(frozen=True, slots=True)
class RefreshTarget:
    kind: str
    view_id: str | None = None

    @classmethod
    def controls(cls) -> "RefreshTarget":
        return cls("controls")

    @classmethod
    def line_plot(cls, view_id: str) -> "RefreshTarget":
        return cls("line_plot", view_id)

    @classmethod
    def morphology(cls, view_id: str) -> "RefreshTarget":
        return cls("morphology", view_id)

    @classmethod
    def surface_visual(cls, view_id: str) -> "RefreshTarget":
        return cls("surface_visual", view_id)

    @classmethod
    def surface_style(cls, view_id: str) -> "RefreshTarget":
        return cls("surface_style", view_id)

    @classmethod
    def surface_axes_geometry(cls, view_id: str) -> "RefreshTarget":
        return cls("surface_axes_geometry", view_id)

    @classmethod
    def surface_axes_style(cls, view_id: str) -> "RefreshTarget":
        return cls("surface_axes_style", view_id)

    @classmethod
    def operator_overlay(cls, view_id: str) -> "RefreshTarget":
        return cls("operator_overlay", view_id)


RefreshTarget.CONTROLS = RefreshTarget.controls()


class RefreshPlanner:
    SURFACE_VISUAL_PROPS = frozenset({"field_id", "geometry_id"})
    SURFACE_STYLE_PROPS = frozenset(
        {
            "color_map",
            "color_limits",
            "color_by",
            "surface_color",
            "surface_shading",
            "surface_alpha",
            "background_color",
        }
    )
    SURFACE_AXES_GEOMETRY_PROPS = frozenset(
        {
            "field_id",
            "geometry_id",
            "render_axes",
            "axes_in_middle",
            "tick_count",
            "tick_length_scale",
            "axis_labels",
        }
    )
    SURFACE_AXES_STYLE_PROPS = frozenset(
        {
            "tick_label_size",
            "axis_label_size",
            "axis_color",
            "text_color",
            "axis_alpha",
        }
    )
    GRID_SLICE_COMPUTE_PROPS = frozenset({"field_id", "geometry_id", "axis_state_key", "position_state_key"})
    GRID_SLICE_STYLE_PROPS = frozenset({"color", "alpha", "fill_alpha", "width"})
    LINE_PLOT_PROPS = frozenset(
        {
            "field_id",
            "operator_id",
            "x_dim",
            "series_dim",
            "selectors",
            "x_label",
            "y_label",
            "x_unit",
            "y_unit",
            "pen",
            "background_color",
            "title",
            "show_legend",
            "series_colors",
            "series_palette",
            "rolling_window",
            "trim_to_rolling_window",
            "y_min",
            "y_max",
            "x_major_tick_spacing",
            "x_minor_tick_spacing",
        }
    )

    def __init__(self, scene: Scene):
        self.scene = scene

    def _view_3d_ids(self) -> tuple[str, ...]:
        return self.scene.layout.resolved_3d_view_ids()

    def _view_3d(self, view_id: str):
        return self.scene.views.get(view_id)

    def morphology_views(self) -> dict[str, MorphologyViewSpec]:
        return {
            view_id: view
            for view_id in self._view_3d_ids()
            if isinstance((view := self._view_3d(view_id)), MorphologyViewSpec)
        }

    def surface_views(self) -> dict[str, SurfaceViewSpec]:
        return {
            view_id: view
            for view_id in self._view_3d_ids()
            if isinstance((view := self._view_3d(view_id)), SurfaceViewSpec)
        }

    def _host_for_view(self, view_id: str) -> View3DHostSpec | None:
        for host in self.scene.layout.view_3d_hosts:
            if view_id in host.view_ids:
                return host
        return None

    def grid_slice_operators_for_view(self, view_id: str) -> dict[str, GridSliceOperatorSpec]:
        surface_view = self.surface_views().get(view_id)
        if surface_view is None:
            return {}
        host = self._host_for_view(view_id)
        if host is None:
            return {}
        operators: dict[str, GridSliceOperatorSpec] = {}
        for operator_id in host.operator_ids:
            operator = self.scene.operators.get(operator_id)
            if not isinstance(operator, GridSliceOperatorSpec):
                continue
            if operator.field_id != surface_view.field_id:
                continue
            if operator.geometry_id not in {None, surface_view.geometry_id}:
                continue
            operators[operator_id] = operator
        return operators

    def line_views(self) -> dict[str, LinePlotViewSpec]:
        return {
            view_id: view
            for view_id in self.scene.layout.resolved_line_plot_view_ids()
            if isinstance((view := self.scene.views.get(view_id)), LinePlotViewSpec)
        }

    def full_refresh_targets(self) -> set[RefreshTarget]:
        targets = {RefreshTarget.CONTROLS}
        for view_id in self.morphology_views():
            targets.add(RefreshTarget.morphology(view_id))
        for view_id in self.surface_views():
            targets.update(
                {
                    RefreshTarget.surface_visual(view_id),
                    RefreshTarget.surface_axes_geometry(view_id),
                    RefreshTarget.operator_overlay(view_id),
                }
            )
        for view_id in self.line_views():
            targets.add(RefreshTarget.line_plot(view_id))
        return targets

    def targets_for_state_change(self, state_key: str) -> set[RefreshTarget]:
        targets: set[RefreshTarget] = set()

        for view_id, morph_view in self.morphology_views().items():
            if binding_key(morph_view.background_color) == state_key or binding_key(morph_view.color_limits) == state_key:
                targets.add(RefreshTarget.morphology(view_id))

        for view_id, surface_view in self.surface_views().items():
            if any(binding_key(getattr(surface_view, prop)) == state_key for prop in self.SURFACE_VISUAL_PROPS if hasattr(surface_view, prop)):
                targets.add(RefreshTarget.surface_visual(view_id))
            if any(binding_key(getattr(surface_view, prop)) == state_key for prop in self.SURFACE_STYLE_PROPS if hasattr(surface_view, prop)):
                targets.add(RefreshTarget.surface_style(view_id))
            if any(
                binding_key(getattr(surface_view, prop)) == state_key
                for prop in self.SURFACE_AXES_GEOMETRY_PROPS
                if hasattr(surface_view, prop)
            ):
                targets.add(RefreshTarget.surface_axes_geometry(view_id))
            if any(
                binding_key(getattr(surface_view, prop)) == state_key
                for prop in self.SURFACE_AXES_STYLE_PROPS
                if hasattr(surface_view, prop)
            ):
                targets.add(RefreshTarget.surface_axes_style(view_id))
            operators = self.grid_slice_operators_for_view(view_id)
            if any(
                binding_key(getattr(operator, prop)) == state_key
                for operator in operators.values()
                for prop in self.GRID_SLICE_STYLE_PROPS
            ):
                targets.add(RefreshTarget.operator_overlay(view_id))
            if any(state_key in {operator.axis_state_key, operator.position_state_key} for operator in operators.values()):
                targets.add(RefreshTarget.operator_overlay(view_id))

        for view_id, line_view in self.line_views().items():
            if line_view.operator_id is not None:
                operator = self.scene.operators.get(line_view.operator_id)
                if isinstance(operator, GridSliceOperatorSpec) and state_key in {operator.axis_state_key, operator.position_state_key}:
                    targets.add(RefreshTarget.line_plot(view_id))
            if any(binding_key(value) == state_key for value in line_view.selectors.values()):
                targets.add(RefreshTarget.line_plot(view_id))
            if any(binding_key(getattr(line_view, prop)) == state_key for prop in ("pen", "background_color")):
                targets.add(RefreshTarget.line_plot(view_id))

        return targets

    def targets_for_field_replace(self, field_id: str, coords_changed: bool = True) -> set[RefreshTarget]:
        targets: set[RefreshTarget] = set()

        for view_id, morph_view in self.morphology_views().items():
            if morph_view.color_field_id == field_id:
                targets.add(RefreshTarget.morphology(view_id))

        for view_id, surface_view in self.surface_views().items():
            if surface_view.field_id != field_id:
                if any(operator.field_id == field_id for operator in self.grid_slice_operators_for_view(view_id).values()):
                    targets.add(RefreshTarget.operator_overlay(view_id))
                continue
            targets.add(RefreshTarget.surface_visual(view_id))
            if coords_changed or surface_view.color_limits is None:
                targets.add(RefreshTarget.surface_axes_geometry(view_id))
            if self.grid_slice_operators_for_view(view_id):
                targets.add(RefreshTarget.operator_overlay(view_id))

        for view_id, line_view in self.line_views().items():
            if line_view.operator_id is not None:
                operator = self.scene.operators.get(line_view.operator_id)
                if isinstance(operator, GridSliceOperatorSpec) and operator.field_id == field_id:
                    targets.add(RefreshTarget.line_plot(view_id))
            elif getattr(line_view, "field_id", None) == field_id:
                targets.add(RefreshTarget.line_plot(view_id))
        return targets

    def targets_for_view_patch(self, view_id: str, changed_props: set[str]) -> set[RefreshTarget]:
        view = self.scene.views.get(view_id)
        if isinstance(view, MorphologyViewSpec):
            return {RefreshTarget.morphology(view_id)}
        if isinstance(view, SurfaceViewSpec):
            targets: set[RefreshTarget] = set()
            if changed_props & self.SURFACE_VISUAL_PROPS:
                targets.add(RefreshTarget.surface_visual(view_id))
            if changed_props & self.SURFACE_STYLE_PROPS:
                targets.add(RefreshTarget.surface_style(view_id))
            if changed_props & self.SURFACE_AXES_GEOMETRY_PROPS:
                targets.add(RefreshTarget.surface_axes_geometry(view_id))
            if changed_props & self.SURFACE_AXES_STYLE_PROPS:
                targets.add(RefreshTarget.surface_axes_style(view_id))
            if changed_props & {"field_id", "geometry_id"}:
                targets.add(RefreshTarget.operator_overlay(view_id))
            return targets
        if isinstance(view, LinePlotViewSpec) and changed_props & self.LINE_PLOT_PROPS:
            return {RefreshTarget.line_plot(view_id)}
        return set()

    def targets_for_operator_patch(self, operator_id: str, changed_props: set[str]) -> set[RefreshTarget]:
        targets: set[RefreshTarget] = set()
        for view_id in self.surface_views():
            if operator_id in self.grid_slice_operators_for_view(view_id):
                targets.add(RefreshTarget.operator_overlay(view_id))
        for view_id, line_view in self.line_views().items():
            if line_view.operator_id == operator_id and changed_props & self.GRID_SLICE_COMPUTE_PROPS:
                targets.add(RefreshTarget.line_plot(view_id))
        return targets


class VispyFrontendWindow(QtWidgets.QMainWindow):
    def __init__(self, app_spec: AppSpec):
        super().__init__()
        self.app_spec = app_spec
        initial_scene = app_spec.scene
        if initial_scene is None and app_spec.session is not None:
            initial_scene = resolve_startup_scene_source(app_spec.session)
        self.scene: Scene | None = None
        self.state: dict[str, Any] = {}
        self.transport: PipeTransport | None = None
        self.refresh_planner: RefreshPlanner | None = None
        self._active_selection_action_id: str | None = None
        if app_spec.interaction_target is not None:
            self.interaction_target = resolve_interaction_target_source(app_spec.interaction_target)
        else:
            self.interaction_target = None

        self.viewports: dict[str, Viewport3DPanel] = {}
        self.view_hosts: dict[str, IndependentCanvas3DHostPanel] = {}
        self._view_to_host_id: dict[str, str] = {}
        self.line_plot_host_panels: dict[str, LinePlotHostPanel] = {}
        self.line_plot_panels: dict[str, LinePlotPanel] = {}
        self.controls_panel = ControlsPanel(self._on_control_changed, self._on_action_invoked)
        self.controls_host = ControlsHostPanel(self.controls_panel)

        self._layout_splitter: QtWidgets.QSplitter | None = None

        self._loading_label = QtWidgets.QLabel("Loading visualization...")
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._stack = QtWidgets.QStackedWidget(self)
        self._stack.addWidget(self._loading_label)

        self.setCentralWidget(self._stack)
        self.resize(1280, 720)
        self.statusBar().showMessage("Starting CompNeuroVis")
        self._show_loading_state()

        if initial_scene is not None:
            self._set_scene(initial_scene)

        if app_spec.session is not None:
            configure_multiprocessing()
            self.transport = PipeTransport(app_spec.session, parent=self)
            self.transport.start()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll_transport)
        self.timer.start(1000 // 60)

    @property
    def viewport(self) -> Viewport3DPanel | None:
        return next(iter(self.viewports.values()), None)

    def line_plot_panel(self, view_id: str) -> LinePlotPanel | None:
        return self.line_plot_panels.get(view_id)

    def viewport_for(self, view_id: str) -> Viewport3DPanel | None:
        return self.viewports.get(view_id)

    def _show_loading_state(self, message: str = "Loading visualization...") -> None:
        self._loading_label.setText(message)
        self._stack.setCurrentWidget(self._loading_label)

    def _show_content_state(self) -> None:
        if self._layout_splitter is not None:
            self._stack.setCurrentWidget(self._layout_splitter)

    def _set_scene(self, scene: Scene) -> None:
        self.scene = scene
        self.refresh_planner = RefreshPlanner(scene)
        self._active_selection_action_id = None
        self.setWindowTitle(self.app_spec.title or scene.layout.title)
        for control in scene.controls.values():
            self.state.setdefault(control.resolved_state_key(), control.default)

        self._rebuild_panels()

        self._update_panel_visibility()
        self._apply_refresh_targets(self.refresh_planner.full_refresh_targets())
        self._show_content_state()

    def _view_3d_ids(self) -> tuple[str, ...]:
        if self.scene is None:
            return ()
        return self.scene.layout.resolved_3d_view_ids()

    def _create_view_host(self, host: View3DHostSpec):
        if host.kind != "independent_canvas":
            raise ValueError(f"Unsupported 3D host kind '{host.kind}'")
        if len(host.view_ids) != 1:
            raise ValueError(
                f"3D host '{host.id}' with kind='independent_canvas' must contain exactly one view id"
            )
        view_id = host.view_ids[0]
        view = self.scene.views.get(view_id) if self.scene is not None else None
        title = host.title or getattr(view, "title", None) or view_id
        return IndependentCanvas3DHostPanel(
            host=host,
            title=title,
            on_entity_selected=self._on_entity_selected,
        )

    def _rebuild_panels(self) -> None:
        self.viewports.clear()
        self.view_hosts.clear()
        self._view_to_host_id.clear()
        self.line_plot_host_panels.clear()
        self.line_plot_panels.clear()

        if self._layout_splitter is not None:
            idx = self._stack.indexOf(self._layout_splitter)
            if idx >= 0:
                self._stack.removeWidget(self._layout_splitter)
            self._layout_splitter.deleteLater()
            self._layout_splitter = None

        outer = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        outer.setChildrenCollapsible(False)
        outer.setOpaqueResize(False)
        self._layout_splitter = outer

        controls_placed = False
        for row_cells in self._resolved_panel_grid():
            if len(row_cells) == 1:
                cell = row_cells[0]
                if cell == "controls":
                    outer.addWidget(self.controls_host)
                    controls_placed = True
                else:
                    widget = self._make_panel_for_cell(cell)
                    if widget is not None:
                        outer.addWidget(widget)
            else:
                row = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
                row.setChildrenCollapsible(False)
                row.setOpaqueResize(False)
                for cell in row_cells:
                    if cell == "controls":
                        row.addWidget(self.controls_host)
                        controls_placed = True
                    else:
                        widget = self._make_panel_for_cell(cell)
                        if widget is not None:
                            row.addWidget(widget)
                outer.addWidget(row)

        if not controls_placed:
            controls, actions = self._resolved_controls_and_actions()
            if controls or actions:
                outer.addWidget(self.controls_host)

        self._stack.addWidget(outer)

    def _resolved_panel_grid(self) -> tuple[tuple[str, ...], ...]:
        if self.scene is None:
            return ()
        grid = self.scene.layout.panel_grid
        if grid:
            return grid
        return self._auto_panel_grid()

    def _auto_panel_grid(self) -> tuple[tuple[str, ...], ...]:
        if self.scene is None:
            return ()
        layout = self.scene.layout
        view_ids = list(layout.resolved_3d_view_ids()) + list(layout.resolved_line_plot_view_ids())
        controls, actions = self._resolved_controls_and_actions()
        has_controls = bool(controls or actions)
        if not view_ids and not has_controls:
            return ()
        rows: list[tuple[str, ...]] = []
        if view_ids:
            rows.append(tuple(view_ids))
        if has_controls:
            rows.append(("controls",))
        return tuple(rows)

    def _make_panel_for_cell(self, cell_id: str) -> QtWidgets.QWidget | None:
        if self.scene is None:
            return None
        view = self.scene.views.get(cell_id)
        if isinstance(view, LinePlotViewSpec):
            host = LinePlotHostPanel(view_id=cell_id, title=view.title or cell_id)
            self.line_plot_host_panels[cell_id] = host
            self.line_plot_panels[cell_id] = host.line_plot_panel
            return host
        if isinstance(view, (MorphologyViewSpec, SurfaceViewSpec)):
            host_spec = next(
                (spec for spec in self.scene.layout.view_3d_hosts if cell_id in spec.view_ids),
                View3DHostSpec(id=cell_id, view_ids=(cell_id,), kind="independent_canvas"),
            )
            panel = self._create_view_host(host_spec)
            self.view_hosts[host_spec.id] = panel
            self.viewports[cell_id] = panel.viewport
            self._view_to_host_id[cell_id] = host_spec.id
            return panel
        return None

    def _morphology_view(self, view_id: str):
        if self.scene is None:
            return None
        view = self.scene.views.get(view_id)
        return view if isinstance(view, MorphologyViewSpec) else None

    def _surface_view(self, view_id: str):
        if self.scene is None:
            return None
        view = self.scene.views.get(view_id)
        return view if isinstance(view, SurfaceViewSpec) else None

    def _line_view(self, view_id: str):
        if self.scene is None:
            return None
        view = self.scene.views.get(view_id)
        return view if isinstance(view, LinePlotViewSpec) else None

    def _view_host(self, view_id: str):
        host_id = self._view_to_host_id.get(view_id)
        if host_id is None:
            return None
        return self.view_hosts.get(host_id)

    def _resolved_controls_and_actions(self) -> tuple[list[ControlSpec], list[ActionSpec]]:
        if self.scene is None:
            return [], []
        controls = [self.scene.controls[control_id] for control_id in self.scene.layout.control_ids if control_id in self.scene.controls]
        actions = [self.scene.actions[action_id] for action_id in self.scene.layout.action_ids if action_id in self.scene.actions]
        return controls, actions

    def _update_panel_visibility(self) -> None:
        controls, actions = self._resolved_controls_and_actions()
        self.controls_host.setVisible(bool(controls or actions))
        self._apply_panel_sizes()

    def _apply_panel_sizes(self) -> None:
        if self._layout_splitter is None:
            return
        width = max(self.width(), 1)
        height = max(self.height(), 1)
        n_rows = self._layout_splitter.count()
        if n_rows == 0:
            return
        last_is_controls = self._layout_splitter.widget(n_rows - 1) is self.controls_host
        if last_is_controls and n_rows > 1:
            ctrl_h = max(1, int(height * 0.35))
            view_h = max(1, int((height - ctrl_h) / (n_rows - 1)))
            sizes = [view_h] * (n_rows - 1) + [ctrl_h]
        else:
            row_h = max(1, int(height / n_rows))
            sizes = [row_h] * n_rows
        self._layout_splitter.setSizes(sizes)
        for i in range(n_rows):
            row_widget = self._layout_splitter.widget(i)
            if isinstance(row_widget, QtWidgets.QSplitter):
                n_cols = row_widget.count()
                if n_cols:
                    row_widget.setSizes([max(1, int(width / n_cols))] * n_cols)

    def _refresh_controls(self) -> None:
        if self.scene is None:
            return
        controls, actions = self._resolved_controls_and_actions()
        self.controls_host.set_section_title(has_controls=bool(controls), has_actions=bool(actions))
        self.controls_panel.set_controls(controls, actions, self.state)

    def _resolved_morphology_state(self, view: MorphologyViewSpec) -> dict[str, Any]:
        return {
            f"{view.id}:background_color": resolve_value(view.background_color, self.state),
            f"{view.id}:color_limits": resolve_value(view.color_limits, self.state),
            f"{view.id}:color_norm": view.color_norm,
        }

    def _resolved_surface_state(self, view: SurfaceViewSpec) -> dict[str, Any]:
        return {
            f"{view.id}:color_map": resolve_value(view.color_map, self.state),
            f"{view.id}:color_limits": resolve_value(view.color_limits, self.state),
            f"{view.id}:color_by": resolve_value(view.color_by, self.state),
            f"{view.id}:surface_color": resolve_value(view.surface_color, self.state),
            f"{view.id}:surface_shading": resolve_value(view.surface_shading, self.state),
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
        }

    def _resolved_grid_slice_state(self, operator: GridSliceOperatorSpec) -> dict[str, Any]:
        resolved = {
            f"{operator.id}:color": resolve_value(operator.color, self.state),
            f"{operator.id}:alpha": resolve_value(operator.alpha, self.state),
            f"{operator.id}:fill_alpha": resolve_value(operator.fill_alpha, self.state),
            f"{operator.id}:width": resolve_value(operator.width, self.state),
        }
        if operator.axis_state_key is not None:
            resolved[operator.axis_state_key] = self.state.get(operator.axis_state_key)
        if operator.position_state_key is not None:
            resolved[operator.position_state_key] = self.state.get(operator.position_state_key)
        return resolved

    def _view_host_spec(self, view_id: str) -> View3DHostSpec | None:
        if self.scene is None:
            return None
        host_id = self._view_to_host_id.get(view_id)
        if host_id is None:
            return None
        for host in self.scene.layout.view_3d_hosts:
            if host.id == host_id:
                return host
        return None

    def _grid_slice_operators_for_view(self, view_id: str) -> list[GridSliceOperatorSpec]:
        if self.scene is None:
            return []
        surface_view = self._surface_view(view_id)
        if surface_view is None:
            return []
        host = self._view_host_spec(view_id)
        if host is None:
            return []
        operators: list[GridSliceOperatorSpec] = []
        for operator_id in host.operator_ids:
            operator = self.scene.operators.get(operator_id)
            if not isinstance(operator, GridSliceOperatorSpec):
                continue
            if operator.field_id != surface_view.field_id:
                continue
            if operator.geometry_id not in {None, surface_view.geometry_id}:
                continue
            operators.append(operator)
        return operators

    def _refresh_morphology(self, view_id: str) -> None:
        if self.scene is None:
            return
        host = self._view_host(view_id)
        if host is None:
            return
        morph_view = self._morphology_view(view_id)
        morphology_geometry = None
        morphology_colors = None
        if morph_view is not None:
            geometry = self.scene.geometries.get(morph_view.geometry_id)
            if isinstance(geometry, MorphologyGeometry):
                morphology_geometry = geometry
                if morph_view.color_field_id:
                    field = self.scene.fields[morph_view.color_field_id]
                    if morph_view.sample_dim and morph_view.sample_dim in field.dims:
                        latest = field.select({morph_view.sample_dim: -1})
                        morphology_colors = latest.values
                    else:
                        morphology_colors = field.values
        host.refresh_morphology(
            view_id=view_id,
            morphology_geometry=morphology_geometry,
            morphology_view=morph_view,
            morphology_colors=morphology_colors,
            resolved_state=self._resolved_morphology_state(morph_view) if morph_view is not None else {},
        )

    def _refresh_surface_visual(self, view_id: str) -> None:
        if self.scene is None:
            return
        host = self._view_host(view_id)
        if host is None:
            return
        surface_view = self._surface_view(view_id)
        surface_field = None
        grid_geometry = None
        if surface_view is not None:
            surface_field = self.scene.fields[surface_view.field_id]
            if surface_view.geometry_id is not None:
                grid_geometry = self.scene.geometries.get(surface_view.geometry_id)
        host.refresh_surface_visual(
            view_id=view_id,
            surface_view=surface_view,
            surface_field=surface_field,
            grid_geometry=grid_geometry,
            resolved_state=self._resolved_surface_state(surface_view) if surface_view is not None else {},
        )

    def _refresh_surface_axes_geometry(self, view_id: str) -> None:
        host = self._view_host(view_id)
        if host is None:
            return
        surface_view = self._surface_view(view_id)
        host.refresh_surface_axes_geometry(
            view_id=view_id,
            surface_view=surface_view,
            resolved_state=self._resolved_surface_state(surface_view) if surface_view is not None else {},
        )

    def _refresh_surface_axes_style(self, view_id: str) -> None:
        host = self._view_host(view_id)
        if host is None:
            return
        surface_view = self._surface_view(view_id)
        host.refresh_surface_axes_style(
            view_id=view_id,
            surface_view=surface_view,
            resolved_state=self._resolved_surface_state(surface_view) if surface_view is not None else {},
        )

    def _refresh_surface_style(self, view_id: str) -> None:
        host = self._view_host(view_id)
        if host is None:
            return
        surface_view = self._surface_view(view_id)
        host.refresh_surface_style(
            view_id=view_id,
            surface_view=surface_view,
            resolved_state=self._resolved_surface_state(surface_view) if surface_view is not None else {},
        )

    def _refresh_operator_overlays(self, view_id: str) -> None:
        host = self._view_host(view_id)
        if host is None:
            return
        surface_view = self._surface_view(view_id)
        operators = self._grid_slice_operators_for_view(view_id)
        host.refresh_operator_overlays(
            view_id=view_id,
            surface_view=surface_view,
            operators=operators,
            resolved_operator_states={operator.id: self._resolved_grid_slice_state(operator) for operator in operators},
        )

    def _refresh_line_plot(self, view_id: str) -> None:
        if self.scene is None:
            return
        host = self.line_plot_host_panels.get(view_id)
        if host is None:
            return
        line_view = self._line_view(view_id)
        geometry_lookup = {
            key: value
            for key, value in self.scene.geometries.items()
            if isinstance(value, MorphologyGeometry)
        }
        line_field = None
        if line_view is not None:
            if line_view.operator_id is not None:
                operator = self.scene.operators.get(line_view.operator_id)
                if isinstance(operator, GridSliceOperatorSpec):
                    line_field = self.scene.fields.get(operator.field_id)
            else:
                line_field = self.scene.fields.get(line_view.field_id)
        host.refresh(line_view, line_field, self.state, geometry_lookup, self.scene.operators)

    def _apply_refresh_targets(self, targets: set[RefreshTarget]) -> None:
        if not targets:
            return

        if RefreshTarget.CONTROLS in targets:
            self._refresh_controls()

        dirty_viewports: set[str] = set()
        for target in sorted(
            (target for target in targets if target.kind == "line_plot" and target.view_id is not None),
            key=lambda target: target.view_id or "",
        ):
            self._refresh_line_plot(target.view_id)
        ordered_targets = sorted(
            (target for target in targets if target.kind not in {"controls", "line_plot"}),
            key=lambda target: (
                {
                    "morphology": 0,
                    "surface_visual": 1,
                    "surface_style": 2,
                    "surface_axes_geometry": 3,
                    "surface_axes_style": 4,
                    "operator_overlay": 5,
                }.get(target.kind, 99),
                target.view_id or "",
            ),
        )
        for target in ordered_targets:
            if target.kind == "morphology" and target.view_id is not None:
                self._refresh_morphology(target.view_id)
                dirty_viewports.add(target.view_id)
            elif target.kind == "surface_visual" and target.view_id is not None:
                self._refresh_surface_visual(target.view_id)
                dirty_viewports.add(target.view_id)
            elif target.kind == "surface_style" and target.view_id is not None:
                self._refresh_surface_style(target.view_id)
                dirty_viewports.add(target.view_id)
            elif target.kind == "surface_axes_geometry" and target.view_id is not None:
                self._refresh_surface_axes_geometry(target.view_id)
                dirty_viewports.add(target.view_id)
            elif target.kind == "surface_axes_style" and target.view_id is not None:
                self._refresh_surface_axes_style(target.view_id)
                dirty_viewports.add(target.view_id)
            elif target.kind == "operator_overlay" and target.view_id is not None:
                self._refresh_operator_overlays(target.view_id)
                dirty_viewports.add(target.view_id)
        dirty_hosts = {self._view_to_host_id[view_id] for view_id in dirty_viewports if view_id in self._view_to_host_id}
        for host_id in dirty_hosts:
            host = self.view_hosts.get(host_id)
            if host is not None:
                host.commit()

    def _poll_transport(self) -> None:
        if self.transport is None:
            return
        pending_targets: set[RefreshTarget] = set()
        pending_status: str | None = None
        pending_field_appends: dict[str, FieldAppend] = {}

        def flush_pending_field_appends() -> None:
            nonlocal pending_targets
            if not pending_field_appends:
                return
            if self.scene is None:
                pending_field_appends.clear()
                return
            for field_id, update in pending_field_appends.items():
                current = self.scene.fields[field_id]
                self.scene.fields[field_id] = current.append(
                    update.append_dim,
                    update.values,
                    update.coord_values,
                    max_length=update.max_length,
                    attrs_update=update.attrs_update,
                )
                if self.refresh_planner is not None:
                    pending_targets.update(self.refresh_planner.targets_for_field_replace(field_id))
            pending_field_appends.clear()

        for update in self.transport.poll_updates():
            if isinstance(update, FieldAppend):
                if self.scene is None:
                    continue
                pending = pending_field_appends.get(update.field_id)
                if pending is None:
                    pending_field_appends[update.field_id] = update
                    continue
                if pending.append_dim != update.append_dim or pending.max_length != update.max_length:
                    flush_pending_field_appends()
                    pending_field_appends[update.field_id] = update
                    continue
                axis = self.scene.fields[update.field_id].axis_index(update.append_dim)
                pending_field_appends[update.field_id] = FieldAppend(
                    field_id=update.field_id,
                    append_dim=update.append_dim,
                    values=np.concatenate([pending.values, update.values], axis=axis),
                    coord_values=np.concatenate([pending.coord_values, update.coord_values], axis=0),
                    max_length=update.max_length,
                    attrs_update={**pending.attrs_update, **update.attrs_update},
                )
                continue

            flush_pending_field_appends()
            if isinstance(update, SceneReady):
                self._set_scene(update.scene)
                pending_targets.clear()
                pending_status = "Scene ready"
            elif isinstance(update, FieldReplace):
                if self.scene is None:
                    continue
                current = self.scene.fields[update.field_id]
                coords_changed = update.coords is not None
                coords = current.coords if update.coords is None else update.coords
                self.scene.fields[update.field_id] = current.with_values(update.values, coords=coords, attrs_update=update.attrs_update)
                if self.refresh_planner is not None:
                    pending_targets.update(self.refresh_planner.targets_for_field_replace(update.field_id, coords_changed=coords_changed))
            elif isinstance(update, ScenePatch):
                if self.scene is None:
                    continue
                for view_id, patch in update.view_updates.items():
                    self.scene.replace_view(view_id, patch)
                    if self.refresh_planner is not None:
                        pending_targets.update(self.refresh_planner.targets_for_view_patch(view_id, set(patch.keys())))
                for operator_id, patch in update.operator_updates.items():
                    self.scene.replace_operator(operator_id, patch)
                    if self.refresh_planner is not None:
                        pending_targets.update(self.refresh_planner.targets_for_operator_patch(operator_id, set(patch.keys())))
                for control_id, patch in update.control_updates.items():
                    self.scene.replace_control(control_id, patch)
                    pending_targets.add(RefreshTarget.CONTROLS)
                self.scene.metadata.update(update.metadata_updates)
            elif isinstance(update, StatePatch):
                if self.refresh_planner is None:
                    continue
                for key, value in update.updates.items():
                    self.state[key] = value
                    pending_targets.update(self.refresh_planner.targets_for_state_change(key))
            elif isinstance(update, Status):
                if update.message:
                    if update.timeout_ms is not None:
                        self.statusBar().showMessage(update.message, update.timeout_ms)
                    else:
                        pending_status = update.message
                else:
                    self.statusBar().clearMessage()
            else:
                msg = getattr(update, "message", str(update))
                pending_status = msg
                sys.stderr.write(f"{msg.rstrip()}\n")
                sys.stderr.flush()
                if getattr(self.transport, "_dead", False):
                    # Worker process died — stop polling and surface the error clearly.
                    self.timer.stop()
                    self.transport = None
                    sys.stderr.write(f"{msg.rstrip()}\n")
                    sys.stderr.flush()
                    QtWidgets.QMessageBox.critical(self, "Session error", msg)
                    return
        flush_pending_field_appends()
        if pending_targets:
            self._apply_refresh_targets(pending_targets)
        if pending_status is not None:
            self.statusBar().showMessage(pending_status)

    def _on_entity_selected(self, entity_id: str) -> None:
        self.state["selected_entity_id"] = entity_id
        if self.scene is not None:
            for geometry in self.scene.geometries.values():
                if isinstance(geometry, MorphologyGeometry) and entity_id in geometry.entity_ids:
                    self.state["selected_entity_label"] = geometry.label_for(entity_id)
                    break
            consumed = self._invoke_interaction_entity_click(entity_id)
            if not consumed and self._active_selection_action_id is not None:
                action = self.scene.actions.get(self._active_selection_action_id)
                if action is not None:
                    payload = {
                        key: resolve_value(value, self.state)
                        for key, value in action.payload.items()
                    }
                    payload[action.selection_payload_key] = entity_id
                    self._send_action(action, payload)
            elif not consumed and self.transport is not None:
                self.transport.send_command(EntityClicked(entity_id))
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
            if action.id == "reset":
                self.transport.send_command(Reset())
            else:
                self.transport.send_command(InvokeAction(action.id, payload))

    def keyPressEvent(self, event) -> None:
        key_text = self._event_key_text(event)
        if key_text and self._invoke_interaction_key_press(key_text):
            event.accept()
            return
        if self.scene is not None:
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
            event.accept()
            return
        if key_text and self.transport is not None:
            self.transport.send_command(KeyPressed(key_text))
            event.accept()
            return
        super().keyPressEvent(event)

    def _action_for_event(self, event: QtGui.QKeyEvent):
        if self.scene is None:
            return None
        pressed = self._event_key_text(event)
        for action_id in self.scene.layout.action_ids:
            action = self.scene.actions.get(action_id)
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
        if target is None:
            return False
        handler = getattr(target, "on_action", None)
        if handler is None:
            return False
        return bool(handler(action_id, payload, self._interaction_context()))

    def _invoke_interaction_key_press(self, key: str) -> bool:
        target = self.interaction_target
        if target is None:
            return False
        handler = getattr(target, "on_key_press", None)
        if handler is None:
            return False
        return bool(handler(key, self._interaction_context()))

    def _invoke_interaction_entity_click(self, entity_id: str) -> bool:
        target = self.interaction_target
        if target is None:
            return False
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
    def scene(self) -> Scene | None:
        return self.window.scene

    @property
    def selected_entity_id(self) -> str | None:
        value = self.window.state.get("selected_entity_id")
        return str(value) if value is not None else None

    def state(self, key: str, default: Any = None) -> Any:
        return self.window.state.get(key, default)

    def entity_info(self, entity_id: str | None = None) -> dict[str, Any] | None:
        current_id = entity_id or self.selected_entity_id
        if current_id is None or self.window.scene is None:
            return None
        for geometry in self.window.scene.geometries.values():
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
        if self.window.scene is None:
            return
        action = self.window.scene.actions.get(action_id)
        if action is None:
            return
        resolved_payload = payload if payload is not None else {
            key: resolve_value(value, self.window.state)
            for key, value in action.payload.items()
        }
        self.window._send_action(action, resolved_payload)

    def set_control(self, control_id: str, value: Any) -> None:
        if self.window.scene is None:
            return
        control = self.window.scene.controls.get(control_id)
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
