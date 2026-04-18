from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
from vispy import scene
from vispy.scene.cameras import TurntableCamera

from compneurovis._perf import perf_log
from compneurovis.core.controls import ActionSpec, ControlSpec
from compneurovis.core.field import Field
from compneurovis.core.geometry import GridGeometry, MorphologyGeometry
from compneurovis.core.operators import GridSliceOperatorSpec, OperatorSpec
from compneurovis.core.scene import PanelSpec
from compneurovis.core.state import StateBinding
from compneurovis.core.views import LinePlotViewSpec, MorphologyViewSpec, SurfaceViewSpec
from compneurovis.frontends.vispy.renderers import MorphologyRenderer, SurfaceRenderer


@dataclass(slots=True)
class SurfaceSceneData:
    field_id: str
    x_dim: str
    y_dim: str
    x_grid: np.ndarray
    y_grid: np.ndarray
    z: np.ndarray
    coords: dict[str, np.ndarray]


class Viewport3DPanel(QtWidgets.QWidget):
    def __init__(
        self,
        *,
        host_spec: PanelSpec | None = None,
        on_entity_selected=None,
        parent=None,
    ):
        super().__init__(parent)
        self.canvas = scene.SceneCanvas(keys="interactive", bgcolor="white", show=False)
        self.view = self.canvas.central_widget.add_view()
        distance = 200.0 if host_spec is None else host_spec.camera_distance
        elevation = 30.0 if host_spec is None else host_spec.camera_elevation
        azimuth = 30.0 if host_spec is None else host_spec.camera_azimuth
        self.view.camera = TurntableCamera(
            fov=60,
            distance=distance,
            elevation=elevation,
            azimuth=azimuth,
            translate_speed=100,
            up="+z",
        )
        self.renderer_morph = MorphologyRenderer(self.view)
        self.renderer_surface = SurfaceRenderer(self.view)
        self.on_entity_selected = on_entity_selected
        self.DRAG_THRESHOLD = 5
        self._mouse_start = None
        self._active_geometry = None
        self._active_mode: str | None = None
        self._surface_scene: SurfaceSceneData | None = None
        self._surface_coord_key: tuple | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas.native)

        self.canvas.events.mouse_press.connect(self._on_mouse_press)
        self.canvas.events.mouse_release.connect(self._on_mouse_release)

    def _on_mouse_press(self, ev):
        self._mouse_start = ev.pos

    def _on_mouse_release(self, ev):
        if self._active_geometry is None or self.on_entity_selected is None or self._mouse_start is None:
            return
        dx = ev.pos[0] - self._mouse_start[0]
        dy = ev.pos[1] - self._mouse_start[1]
        self._mouse_start = None
        if dx * dx + dy * dy > self.DRAG_THRESHOLD ** 2:
            return
        x, y = ev.pos
        _, h = self.canvas.size
        ps = self.canvas.pixel_scale
        xf, yf = int(x * ps), int((h - y - 1) * ps)
        entity_id = self.renderer_morph.pick(xf, yf, self.canvas)
        if entity_id:
            self.on_entity_selected(entity_id)

    def _set_mode(self, mode: str | None) -> None:
        if self._active_mode == mode:
            return
        if mode != "morphology":
            self.renderer_morph.clear()
            self._active_geometry = None
        if mode != "surface":
            self.renderer_surface.clear()
            self._surface_scene = None
            self._surface_coord_key = None
        self._active_mode = mode

    def clear(self) -> None:
        self._set_mode(None)
        self.canvas.native.setVisible(False)

    def refresh_morphology(
        self,
        *,
        morphology_geometry: MorphologyGeometry | None,
        morphology_view: MorphologyViewSpec | None,
        morphology_colors: np.ndarray | None,
        resolved_state: dict[str, Any],
    ) -> None:
        if morphology_view is None or morphology_geometry is None:
            return

        self._set_mode("morphology")
        self.canvas.native.setVisible(True)
        self._active_geometry = morphology_geometry
        self.canvas.bgcolor = resolved_state.get(f"{morphology_view.id}:background_color", morphology_view.background_color)
        if self.renderer_morph.geometry is not morphology_geometry:
            self.renderer_morph.set_geometry(morphology_geometry)
        if morphology_colors is not None:
            self.renderer_morph.update_colors(
                morphology_colors,
                morphology_view.color_map,
                color_limits=resolved_state.get(f"{morphology_view.id}:color_limits", morphology_view.color_limits),
                color_norm=resolved_state.get(f"{morphology_view.id}:color_norm", morphology_view.color_norm),
            )

    def refresh_surface_visual(
        self,
        *,
        surface_view: SurfaceViewSpec | None,
        surface_field: Field | None,
        grid_geometry: GridGeometry | None,
        resolved_state: dict[str, Any],
    ) -> None:
        if surface_view is None or surface_field is None:
            return

        self._set_mode("surface")
        self.canvas.native.setVisible(True)
        self.canvas.bgcolor = resolved_state[f"{surface_view.id}:background_color"]

        # Build a coord key from the grid source. Shape-based identity is sufficient —
        # if shape matches, coordinates are the same for all practical animation use cases.
        if grid_geometry is not None:
            coord_key = (grid_geometry.id,) + tuple(c.shape for c in grid_geometry.coords.values())
        else:
            coord_key = (surface_field.id,) + tuple(c.shape for c in surface_field.coords.values())

        coords_changed = coord_key != self._surface_coord_key
        if coords_changed:
            self._surface_scene = surface_scene_from_field(surface_field, grid_geometry)
            self._surface_coord_key = coord_key
        else:
            # Reuse cached x_grid/y_grid — only update z values.
            z = surface_field.values
            if surface_field.dims != (self._surface_scene.y_dim, self._surface_scene.x_dim):
                axis_map = {dim: idx for idx, dim in enumerate(surface_field.dims)}
                z = np.transpose(z, (axis_map[self._surface_scene.y_dim], axis_map[self._surface_scene.x_dim]))
            self._surface_scene = SurfaceSceneData(
                field_id=self._surface_scene.field_id,
                x_dim=self._surface_scene.x_dim,
                y_dim=self._surface_scene.y_dim,
                x_grid=self._surface_scene.x_grid,
                y_grid=self._surface_scene.y_grid,
                z=np.asarray(z, dtype=np.float32),
                coords=self._surface_scene.coords,
            )

        self.renderer_surface.update_surface(
            self._surface_scene.x_grid,
            self._surface_scene.y_grid,
            self._surface_scene.z,
            color_map=resolved_state[f"{surface_view.id}:color_map"],
            color_limits=resolved_state[f"{surface_view.id}:color_limits"],
            colors=None,
            color_by=resolved_state[f"{surface_view.id}:color_by"],
            surface_color=resolved_state[f"{surface_view.id}:surface_color"],
            surface_shading=resolved_state[f"{surface_view.id}:surface_shading"],
            surface_alpha=resolved_state[f"{surface_view.id}:surface_alpha"],
            coords_changed=coords_changed,
        )

    def refresh_surface_style(
        self,
        *,
        surface_view: SurfaceViewSpec | None,
        resolved_state: dict[str, Any],
    ) -> None:
        if surface_view is None or self._surface_scene is None:
            return

        self._set_mode("surface")
        self.canvas.native.setVisible(True)
        self.canvas.bgcolor = resolved_state[f"{surface_view.id}:background_color"]
        self.renderer_surface.update_surface_style(
            self._surface_scene.z,
            color_map=resolved_state[f"{surface_view.id}:color_map"],
            color_limits=resolved_state[f"{surface_view.id}:color_limits"],
            colors=None,
            color_by=resolved_state[f"{surface_view.id}:color_by"],
            surface_color=resolved_state[f"{surface_view.id}:surface_color"],
            surface_shading=resolved_state[f"{surface_view.id}:surface_shading"],
            surface_alpha=resolved_state[f"{surface_view.id}:surface_alpha"],
        )

    def refresh_surface_axes_geometry(
        self,
        *,
        surface_view: SurfaceViewSpec | None,
        resolved_state: dict[str, Any],
    ) -> None:
        if surface_view is None or self._surface_scene is None:
            self.renderer_surface.axes.clear()
            return

        axis_labels = surface_view.axis_labels or (
            self._surface_scene.x_dim,
            self._surface_scene.y_dim,
            self._surface_scene.field_id,
        )
        self.renderer_surface.axes.set_axes_geometry(
            render_axes=resolved_state[f"{surface_view.id}:render_axes"],
            axes_in_middle=resolved_state[f"{surface_view.id}:axes_in_middle"],
            tick_count=resolved_state[f"{surface_view.id}:tick_count"],
            tick_length_scale=resolved_state[f"{surface_view.id}:tick_length_scale"],
            axis_labels=axis_labels,
            x=self._surface_scene.x_grid,
            y=self._surface_scene.y_grid,
            z=self._surface_scene.z,
        )
        self.renderer_surface.axes.set_axes_style(
            render_axes=resolved_state[f"{surface_view.id}:render_axes"],
            tick_label_size=resolved_state[f"{surface_view.id}:tick_label_size"],
            axis_label_size=resolved_state[f"{surface_view.id}:axis_label_size"],
            axis_color=resolved_state[f"{surface_view.id}:axis_color"],
            text_color=resolved_state[f"{surface_view.id}:text_color"],
            axis_alpha=resolved_state[f"{surface_view.id}:axis_alpha"],
        )

    def refresh_surface_axes_style(
        self,
        *,
        surface_view: SurfaceViewSpec | None,
        resolved_state: dict[str, Any],
    ) -> None:
        if surface_view is None or self._surface_scene is None:
            self.renderer_surface.axes.clear()
            return

        self.renderer_surface.axes.set_axes_style(
            render_axes=resolved_state[f"{surface_view.id}:render_axes"],
            tick_label_size=resolved_state[f"{surface_view.id}:tick_label_size"],
            axis_label_size=resolved_state[f"{surface_view.id}:axis_label_size"],
            axis_color=resolved_state[f"{surface_view.id}:axis_color"],
            text_color=resolved_state[f"{surface_view.id}:text_color"],
            axis_alpha=resolved_state[f"{surface_view.id}:axis_alpha"],
        )

    def refresh_operator_overlays(
        self,
        *,
        surface_view: SurfaceViewSpec | None,
        operators: list[GridSliceOperatorSpec],
        resolved_operator_states: dict[str, dict[str, Any]],
    ) -> None:
        if surface_view is None or self._surface_scene is None or not operators:
            self.renderer_surface.clear_operator_overlays()
            return

        overlays = []
        for operator in operators:
            overlay = overlay_from_grid_slice_operator(
                self._surface_scene,
                operator,
                resolved_operator_states.get(operator.id, {}),
            )
            if overlay is not None:
                overlays.append(overlay)
        if not overlays:
            self.renderer_surface.clear_operator_overlays()
            return
        self.renderer_surface.set_slice_operator_overlays(
            overlays,
            x=self._surface_scene.x_grid,
            y=self._surface_scene.y_grid,
            z=self._surface_scene.z,
        )

    def commit(self) -> None:
        self.canvas.update()


class IndependentCanvas3DHostPanel(QtWidgets.QGroupBox):
    def __init__(self, *, panel: PanelSpec, title: str | None = None, on_entity_selected=None, parent=None):
        super().__init__(title or panel.view_ids[0], parent)
        self.panel_id = panel.id
        self.view_ids = panel.view_ids
        self.viewport = Viewport3DPanel(host_spec=panel, on_entity_selected=on_entity_selected)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 4)
        layout.addWidget(self.viewport)

    def clear(self) -> None:
        self.viewport.clear()

    def refresh_morphology(
        self,
        *,
        view_id: str,
        morphology_geometry: MorphologyGeometry | None,
        morphology_view: MorphologyViewSpec | None,
        morphology_colors: np.ndarray | None,
        resolved_state: dict[str, Any],
    ) -> None:
        if view_id != self.view_ids[0]:
            return
        self.viewport.refresh_morphology(
            morphology_geometry=morphology_geometry,
            morphology_view=morphology_view,
            morphology_colors=morphology_colors,
            resolved_state=resolved_state,
        )

    def refresh_surface_visual(
        self,
        *,
        view_id: str,
        surface_view: SurfaceViewSpec | None,
        surface_field: Field | None,
        grid_geometry: GridGeometry | None,
        resolved_state: dict[str, Any],
    ) -> None:
        if view_id != self.view_ids[0]:
            return
        self.viewport.refresh_surface_visual(
            surface_view=surface_view,
            surface_field=surface_field,
            grid_geometry=grid_geometry,
            resolved_state=resolved_state,
        )

    def refresh_surface_style(
        self,
        *,
        view_id: str,
        surface_view: SurfaceViewSpec | None,
        resolved_state: dict[str, Any],
    ) -> None:
        if view_id != self.view_ids[0]:
            return
        self.viewport.refresh_surface_style(
            surface_view=surface_view,
            resolved_state=resolved_state,
        )

    def refresh_surface_axes_geometry(
        self,
        *,
        view_id: str,
        surface_view: SurfaceViewSpec | None,
        resolved_state: dict[str, Any],
    ) -> None:
        if view_id != self.view_ids[0]:
            return
        self.viewport.refresh_surface_axes_geometry(
            surface_view=surface_view,
            resolved_state=resolved_state,
        )

    def refresh_surface_axes_style(
        self,
        *,
        view_id: str,
        surface_view: SurfaceViewSpec | None,
        resolved_state: dict[str, Any],
    ) -> None:
        if view_id != self.view_ids[0]:
            return
        self.viewport.refresh_surface_axes_style(
            surface_view=surface_view,
            resolved_state=resolved_state,
        )

    def refresh_operator_overlays(
        self,
        *,
        view_id: str,
        surface_view: SurfaceViewSpec | None,
        operators: list[GridSliceOperatorSpec],
        resolved_operator_states: dict[str, dict[str, Any]],
    ) -> None:
        if view_id != self.view_ids[0]:
            return
        self.viewport.refresh_operator_overlays(
            surface_view=surface_view,
            operators=operators,
            resolved_operator_states=resolved_operator_states,
        )

    def commit(self) -> None:
        self.viewport.commit()


class LinePlotPanel(pg.PlotWidget):
    def __init__(self, parent=None, *, show_internal_title: bool = True):
        super().__init__(parent=parent, title="Plot" if show_internal_title else "")
        self._show_internal_title = show_internal_title
        self._resolved_title = ""
        self.setBackground("w")
        self._plot_item = self.plot([], [], pen="k")
        self._series_items: dict[str, pg.PlotDataItem] = {}
        self._legend_signature: tuple[str, ...] | None = None
        # Per-refresh fast-path caches. Each gates one piece of work that does
        # not depend on the data tail. Cleared via _clear_render_caches() when
        # structure changes (view None, _clear_series, mode transitions).
        self._cache_structural_signature: tuple[Any, ...] | None = None
        self._cache_pens: dict[str, tuple[Any, Any]] = {}
        self._cache_y_range_applied: tuple[float | None, float | None] | None = None
        self._cache_x_range_applied: tuple[float, float] | None = None
        self._cache_tick_signature: tuple[Any, ...] | str | None = None
        self._cache_background: Any = None

    @property
    def resolved_title(self) -> str:
        return self._resolved_title

    def _set_resolved_title(self, title: str) -> None:
        self._resolved_title = str(title)
        self.setTitle(self._resolved_title if self._show_internal_title else "")

    def refresh(
        self,
        view: LinePlotViewSpec | None,
        field: Field | None,
        state: dict[str, Any],
        geometry_lookup: dict[str, MorphologyGeometry],
        operator_lookup: dict[str, OperatorSpec] | None = None,
    ) -> None:
        operator_lookup = {} if operator_lookup is None else operator_lookup
        if view is None or field is None:
            self._clear_series()
            self._plot_item.setData([], [])
            self._set_resolved_title("")
            self._reset_view_ranges()
            self._clear_render_caches()
            return

        background = resolve_binding(view.background_color, state)
        if background is not None and background != self._cache_background:
            self.setBackground(background)
            self._cache_background = background

        if view.operator_id is not None:
            self._clear_series()
            operator = operator_lookup.get(view.operator_id)
            if not isinstance(operator, GridSliceOperatorSpec):
                self._plot_item.setData([], [])
                return
            line = line_from_grid_slice_operator(field, operator, state)
            if line is None:
                self._plot_item.setData([], [])
                return
            x, y, x_dim, slice_dim, slice_value = line
            x, y = self._trim_line_data(view, x, y)
            structural_sig = (
                "operator", view.id, x_dim, view.x_unit, view.y_label, view.y_unit,
                view.title, slice_dim, round(float(slice_value), 6),
            )
            if structural_sig != self._cache_structural_signature:
                self.setLabel("bottom", x_dim, view.x_unit)
                self.setLabel("left", view.y_label, view.y_unit)
                title = view.title or field.id
                self._set_resolved_title(f"{title} at {slice_dim} = {slice_value:.3f}")
                self._cache_structural_signature = structural_sig
                self._cache_pens.clear()
            resolved_color = resolve_binding(view.pen, state)
            cached_pen = self._cache_pens.get("__single__")
            if cached_pen is None or cached_pen[0] != resolved_color:
                pen = pg.mkPen(resolved_color, width=2)
                self._cache_pens["__single__"] = (resolved_color, pen)
                self._plot_item.setPen(pen)
            self._plot_item.setData(x, y)
            self._apply_view_ranges(view, x)
            return

        resolved_selectors = {}
        for dim, selector in view.selectors.items():
            resolved = resolve_binding(selector, state)
            if resolved is None:
                self._plot_item.setData([], [])
                return
            filtered = self._filter_selector_for_field(field, dim, resolved)
            if filtered is None:
                self._clear_series()
                self._plot_item.setData([], [])
                return
            resolved_selectors[dim] = filtered

        try:
            sliced = field.select(resolved_selectors)
        except KeyError:
            self._clear_series()
            self._plot_item.setData([], [])
            return
        x_dim = view.x_dim or sliced.dims[-1]
        if view.series_dim is not None:
            self._plot_item.setData([], [])
            self._refresh_series(view, sliced, x_dim, state)
            return

        self._clear_series()
        if len(sliced.dims) != 1 or sliced.dims[0] != x_dim:
            raise ValueError(f"LinePlotViewSpec '{view.id}' must resolve to a 1D field along '{x_dim}'")

        x = np.asarray(sliced.coord(x_dim), dtype=np.float32)
        y = np.asarray(sliced.values, dtype=np.float32)
        x, y = self._trim_line_data(view, x, y)
        title = view.title or field.id
        entity_id = state.get("selected_entity_id")
        entity_label: str | None = None
        if entity_id:
            for geometry in geometry_lookup.values():
                if entity_id in geometry.entity_ids:
                    entity_label = geometry.label_for(entity_id)
                    title = f"{title}: {entity_label}"
                    break
        structural_sig = (
            "single", view.id, view.x_label or x_dim, view.x_unit,
            view.y_label, view.y_unit, title,
        )
        if structural_sig != self._cache_structural_signature:
            self.setLabel("bottom", view.x_label or x_dim, view.x_unit)
            self.setLabel("left", view.y_label, view.y_unit)
            self._set_resolved_title(title)
            self._cache_structural_signature = structural_sig
            self._cache_pens.clear()
        resolved_color = resolve_binding(view.pen, state)
        cached_pen = self._cache_pens.get("__single__")
        if cached_pen is None or cached_pen[0] != resolved_color:
            pen = pg.mkPen(resolved_color, width=2)
            self._cache_pens["__single__"] = (resolved_color, pen)
            self._plot_item.setPen(pen)
        self._plot_item.setData(x, y)
        self._apply_view_ranges(view, x)

    def _filter_selector_for_field(self, field: Field, dim: str, selector: Any) -> Any | None:
        coord = field.coord(dim)
        if isinstance(selector, str):
            return selector if np.any(coord.astype(str) == selector) else None
        if isinstance(selector, (list, tuple, np.ndarray)):
            selector_array = np.asarray(selector)
            if selector_array.ndim != 1 or selector_array.size == 0:
                return None if selector_array.size == 0 else selector
            if np.issubdtype(selector_array.dtype, np.integer) or np.issubdtype(selector_array.dtype, np.floating):
                return selector
            coord_labels = set(coord.astype(str).tolist())
            filtered = [value for value in selector_array.astype(str).tolist() if value in coord_labels]
            return filtered or None
        return selector

    def _ensure_legend(self, enabled: bool) -> None:
        if enabled and self.plotItem.legend is None:
            self.addLegend(offset=(10, 10))
        elif not enabled and self.plotItem.legend is not None:
            self.plotItem.legend.scene().removeItem(self.plotItem.legend)
            self.plotItem.legend = None
            self._legend_signature = None

    def _clear_series(self) -> None:
        if self._series_items:
            for item in self._series_items.values():
                self.removeItem(item)
            self._series_items.clear()
        if self.plotItem.legend is not None:
            self.plotItem.legend.clear()
        self._legend_signature = None

    def _clear_render_caches(self) -> None:
        self._cache_structural_signature = None
        self._cache_pens.clear()
        self._cache_y_range_applied = None
        self._cache_x_range_applied = None
        self._cache_tick_signature = None
        self._cache_background = None

    def _refresh_series(self, view: LinePlotViewSpec, field: Field, x_dim: str, state: dict[str, Any]) -> None:
        series_dim = view.series_dim
        if series_dim is None:
            raise ValueError("series_dim is required for multi-series refresh")
        if set(field.dims) != {series_dim, x_dim} or field.values.ndim != 2:
            raise ValueError(
                f"LinePlotViewSpec '{view.id}' with series_dim='{series_dim}' must resolve to a 2D field over ({series_dim}, {x_dim})"
            )

        axis_map = {dim: idx for idx, dim in enumerate(field.dims)}
        values = field.values
        if values.dtype != np.float32:
            values = np.asarray(values, dtype=np.float32)
        if field.dims != (series_dim, x_dim):
            values = np.transpose(values, axes=(axis_map[series_dim], axis_map[x_dim]))

        x_coord = field.coord(x_dim)
        x = x_coord if x_coord.dtype == np.float32 else np.asarray(x_coord, dtype=np.float32)
        series_labels = [str(label) for label in field.coord(series_dim)]
        x, values = self._trim_series_data(view, x, values)

        title = view.title or field.id
        structural_sig = (
            "series", view.id, view.x_label or x_dim, view.x_unit,
            view.y_label, view.y_unit, title, view.show_legend,
            tuple(series_labels),
        )
        if structural_sig != self._cache_structural_signature:
            self.setLabel("bottom", view.x_label or x_dim, view.x_unit)
            self.setLabel("left", view.y_label, view.y_unit)
            self._set_resolved_title(title)
            self._ensure_legend(view.show_legend)
            self._cache_structural_signature = structural_sig
            self._cache_pens.clear()

        stale = set(self._series_items.keys()) - set(series_labels)
        for label in stale:
            self.removeItem(self._series_items[label])
            del self._series_items[label]
            self._cache_pens.pop(label, None)

        visible_xmin: float | None = None
        visible_xmax: float | None = None
        for idx, label in enumerate(series_labels):
            if label in view.series_colors:
                color = view.series_colors[label]
            elif view.series_palette:
                color = view.series_palette[idx % len(view.series_palette)]
            else:
                color = view.pen
            resolved_color = resolve_binding(color, state)
            cached = self._cache_pens.get(label)
            if cached is None or cached[0] != resolved_color:
                pen = pg.mkPen(resolved_color, width=2)
                self._cache_pens[label] = (resolved_color, pen)
                pen_changed = True
            else:
                pen = cached[1]
                pen_changed = False

            item = self._series_items.get(label)
            if item is None:
                item = self.plot([], [], pen=pen)
                self._series_items[label] = item
            elif pen_changed:
                item.setPen(pen)

            series_x, series_y = self._finite_line_data(x, values[idx])
            item.setData(series_x, series_y)
            if len(series_x):
                series_xmin = float(np.min(series_x))
                series_xmax = float(np.max(series_x))
                visible_xmin = series_xmin if visible_xmin is None else min(visible_xmin, series_xmin)
                visible_xmax = series_xmax if visible_xmax is None else max(visible_xmax, series_xmax)

        if self.plotItem.legend is not None:
            legend_signature = tuple(series_labels)
            if legend_signature != self._legend_signature:
                self.plotItem.legend.clear()
                for label in series_labels:
                    self.plotItem.legend.addItem(self._series_items[label], label)
                self._legend_signature = legend_signature
        else:
            self._legend_signature = None
        if visible_xmin is None or visible_xmax is None:
            range_x = np.asarray([], dtype=np.float32)
        else:
            range_x = np.asarray([visible_xmin, visible_xmax], dtype=np.float32)
        self._apply_view_ranges(view, range_x)

    def _reset_view_ranges(self) -> None:
        vb = self.plotItem.getViewBox()
        vb.enableAutoRange(x=True, y=True)
        vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
        self._reset_tick_spacing()
        self._cache_x_range_applied = None
        self._cache_y_range_applied = None
        self._cache_tick_signature = None

    def _apply_view_ranges(self, view: LinePlotViewSpec, x: np.ndarray) -> None:
        vb = self.plotItem.getViewBox()
        xmin = 0.0
        xmax = 0.0
        data_xmin = 0.0
        data_xmax = 0.0

        if view.y_min is not None or view.y_max is not None:
            y_target = (view.y_min, view.y_max)
            if y_target != self._cache_y_range_applied:
                vb.enableAutoRange(y=False)
                vb.setLimits(yMin=view.y_min, yMax=view.y_max)
                if view.y_min is not None and view.y_max is not None:
                    vb.setYRange(float(view.y_min), float(view.y_max), padding=0)
                self._cache_y_range_applied = y_target
        else:
            if self._cache_y_range_applied is not None:
                vb.enableAutoRange(y=True)
                vb.setLimits(yMin=None, yMax=None)
                self._cache_y_range_applied = None

        if len(x):
            data_xmin = float(np.min(x))
            data_xmax = float(np.max(x))

        if view.rolling_window is not None and len(x):
            xmax = data_xmax
            xmin = max(data_xmin, xmax - float(view.rolling_window))
            if xmax <= xmin:
                applied = (xmin, xmin + max(float(view.rolling_window), 1e-6))
            else:
                applied = (xmin, xmax)
            if applied != self._cache_x_range_applied:
                vb.enableAutoRange(x=False)
                vb.setXRange(applied[0], applied[1], padding=0)
                self._cache_x_range_applied = applied
        else:
            if len(x):
                xmin = data_xmin
                xmax = data_xmax
            if self._cache_x_range_applied is not None:
                vb.enableAutoRange(x=True)
                vb.setLimits(xMin=None, xMax=None)
                self._cache_x_range_applied = None

        self._apply_tick_spacing(view, xmin, xmax)

    def _trim_line_data(self, view: LinePlotViewSpec, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not view.trim_to_rolling_window or view.rolling_window is None or len(x) == 0:
            return self._finite_line_data(x, y)
        mask = self._rolling_window_mask(x, float(view.rolling_window))
        return self._finite_line_data(x[mask], y[mask])

    def _trim_series_data(
        self,
        view: LinePlotViewSpec,
        x: np.ndarray,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not view.trim_to_rolling_window or view.rolling_window is None or len(x) == 0:
            return x, values
        mask = self._rolling_window_mask(x, float(view.rolling_window))
        return x[mask], values[:, mask]

    def _finite_line_data(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = np.isfinite(x) & np.isfinite(y)
        return x[mask], y[mask]

    def _rolling_window_mask(self, x: np.ndarray, window: float) -> np.ndarray:
        xmin = float(x[-1]) - window
        mask = x >= xmin
        if np.any(mask):
            first_visible = int(np.argmax(mask))
            if first_visible > 0:
                # Keep the sample immediately before the window so the plotted line
                # enters at the left boundary instead of appearing after a gap.
                mask[first_visible - 1] = True
        return mask

    def _apply_tick_spacing(self, view: LinePlotViewSpec, xmin: float, xmax: float) -> None:
        axis = self.plotItem.getAxis("bottom")
        if view.x_major_tick_spacing is not None or view.x_minor_tick_spacing is not None:
            major = view.x_major_tick_spacing
            minor = view.x_minor_tick_spacing
            if minor is None and major is not None:
                minor = major / 5.0
            # Tick set changes only when the visible bounds cross the smallest
            # spacing that can add or remove a visible tick.
            signature_spacing = minor if minor is not None and minor > 0 else major
            if signature_spacing and signature_spacing > 0:
                grid_lo = math.floor((xmin - 1e-9) / signature_spacing)
                grid_hi = math.ceil((xmax + 1e-9) / signature_spacing)
            else:
                grid_lo, grid_hi = xmin, xmax
            sig = (major, minor, grid_lo, grid_hi)
            if sig != self._cache_tick_signature:
                axis.setTicks(self._manual_tick_levels(xmin, xmax, major, minor))
                self._cache_tick_signature = sig
        else:
            if self._cache_tick_signature != "auto":
                self._reset_tick_spacing()
                self._cache_tick_signature = "auto"

    def _reset_tick_spacing(self) -> None:
        axis = self.plotItem.getAxis("bottom")
        axis.setTicks(None)
        axis.setTickSpacing()

    def _manual_tick_levels(self, xmin: float, xmax: float, major: float | None, minor: float | None):
        if major is None:
            return None
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        major_ticks = self._build_tick_values(xmin, xmax, major)
        minor_ticks = self._build_tick_values(xmin, xmax, minor) if minor is not None and minor > 0 else []
        major_values = {round(value, 9) for value in major_ticks}
        minor_ticks = [value for value in minor_ticks if round(value, 9) not in major_values]
        return [
            [(value, self._format_tick_label(value, major)) for value in major_ticks],
            [(value, "") for value in minor_ticks],
        ]

    def _build_tick_values(self, xmin: float, xmax: float, spacing: float | None) -> list[float]:
        if spacing is None or spacing <= 0:
            return []
        start = math.ceil((xmin - 1e-9) / spacing) * spacing
        values = []
        value = start
        while value <= xmax + 1e-9:
            values.append(round(value, 9))
            value += spacing
        return values

    def _format_tick_label(self, value: float, spacing: float) -> str:
        if spacing >= 1 and abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        decimals = max(0, min(6, int(math.ceil(-math.log10(spacing))) if spacing < 1 else 0))
        text = f"{value:.{decimals}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text


class LinePlotHostPanel(QtWidgets.QGroupBox):
    def __init__(self, *, panel_id: str, view_id: str, title: str | None = None, parent=None):
        super().__init__(title or view_id, parent)
        self.panel_id = panel_id
        self.view_id = view_id
        self.line_plot_panel = LinePlotPanel(show_internal_title=False)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 4)
        layout.addWidget(self.line_plot_panel)

    def refresh(
        self,
        view: LinePlotViewSpec | None,
        field: Field | None,
        state: dict[str, Any],
        geometry_lookup: dict[str, MorphologyGeometry],
        operator_lookup: dict[str, OperatorSpec] | None = None,
    ) -> None:
        started = time.monotonic()
        self.line_plot_panel.refresh(view, field, state, geometry_lookup, operator_lookup)
        if view is None:
            self.setTitle("")
            return
        title = self.line_plot_panel.resolved_title or view.title or view.id
        self.setTitle(title)
        duration_ms = round((time.monotonic() - started) * 1000.0, 3)
        if duration_ms >= 5.0:
            perf_log(
                "line_plot",
                "refresh",
                panel_id=self.panel_id,
                view_id=self.view_id,
                field_id=getattr(view, "field_id", None),
                duration_ms=duration_ms,
                field_shape=getattr(getattr(field, "values", None), "shape", None),
            )


class ControlsPanel(QtWidgets.QWidget):
    _MULTI_COLUMN_MIN_WIDTH = 900
    _MULTI_COLUMN_MIN_ITEMS = 8

    def __init__(self, on_value_changed, on_action_invoked=None, parent=None):
        super().__init__(parent)
        self.on_value_changed = on_value_changed
        self.on_action_invoked = on_action_invoked
        self.widgets: dict[str, QtWidgets.QWidget] = {}
        self._controls: list[ControlSpec] = []
        self._actions: list[ActionSpec] = []
        self._state: dict[str, Any] = {}
        self._column_count = 1
        self._grid = QtWidgets.QGridLayout(self)
        self._grid.setContentsMargins(6, 6, 6, 6)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(6)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop)

    def set_controls(self, controls: list[ControlSpec], actions: list[ActionSpec], state: dict[str, Any]) -> None:
        self._controls = list(controls)
        self._actions = list(actions)
        self._state = state
        self._rebuild_grid(force=True)

    def _invoke_action(self, action: ActionSpec, state: dict[str, Any]) -> None:
        if self.on_action_invoked is None:
            return
        payload = {
            key: resolve_binding(value, state)
            for key, value in action.payload.items()
        }
        self.on_action_invoked(action, payload)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rebuild_grid(force=False)

    def _desired_column_count(self) -> int:
        item_count = len(self._controls) + len(self._actions)
        if item_count < self._MULTI_COLUMN_MIN_ITEMS:
            return 1
        if self.width() < self._MULTI_COLUMN_MIN_WIDTH:
            return 1
        return 2

    def _clear_grid(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _rebuild_grid(self, *, force: bool) -> None:
        column_count = self._desired_column_count()
        if not force and column_count == self._column_count:
            return

        self._column_count = column_count
        self._clear_grid()
        self.widgets.clear()

        for column in range(column_count):
            self._grid.setColumnStretch(column, 1)

        row_index = 0
        for index, control in enumerate(self._controls):
            row = row_index + (index // column_count)
            column = index % column_count
            self._grid.addWidget(self._build_control_row(control, self._state), row, column)
        if self._controls:
            row_index += math.ceil(len(self._controls) / column_count)

        if self._controls and self._actions:
            divider = QtWidgets.QFrame()
            divider.setFrameShape(QtWidgets.QFrame.Shape.HLine)
            divider.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
            self._grid.addWidget(divider, row_index, 0, 1, column_count)
            row_index += 1

        for index, action in enumerate(self._actions):
            row = row_index + (index // column_count)
            column = index % column_count
            self._grid.addWidget(self._build_action_button(action, self._state), row, column)

        if self._actions:
            row_index += math.ceil(len(self._actions) / column_count)

        self._grid.setRowStretch(row_index, 1)

    def _build_control_row(self, control: ControlSpec, state: dict[str, Any]) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(QtWidgets.QLabel(control.label))
        key = control.resolved_state_key()
        current = state.get(key, control.default)

        if control.kind in ("float", "double"):
            slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
            steps = int(control.steps or 100)
            slider.setRange(0, steps)
            mn = float(control.min if control.min is not None else 0.0)
            mx = float(control.max if control.max is not None else 1.0)
            value_label = QtWidgets.QLabel("")

            def on_change(raw: int, *, spec=control, label=value_label, min_value=mn, max_value=mx, steps_value=steps):
                frac = raw / max(1, steps_value)
                if spec.scale == "log" and min_value > 0 and max_value > min_value:
                    value = float(min_value * ((max_value / min_value) ** frac))
                else:
                    value = float(min_value + (max_value - min_value) * frac)
                label.setText(f"{value:.3g}")
                self.on_value_changed(spec, value)

            try:
                v0 = int(round((float(current) - mn) / (mx - mn) * steps)) if mx > mn else 0
            except Exception:
                v0 = 0
            slider.setValue(max(0, min(steps, v0)))
            slider.valueChanged.connect(on_change)
            frac = slider.value() / max(1, steps)
            if control.scale == "log" and mn > 0 and mx > mn:
                initial_value = float(mn * ((mx / mn) ** frac))
            else:
                initial_value = float(mn + (mx - mn) * frac)
            value_label.setText(f"{initial_value:.3g}")
            row_layout.addWidget(slider, 1)
            row_layout.addWidget(value_label)
            self.widgets[control.id] = slider

        elif control.kind == "int":
            spin = QtWidgets.QSpinBox()
            spin.setRange(int(control.min if control.min is not None else 0), int(control.max if control.max is not None else 100))
            spin.setValue(int(current))
            spin.valueChanged.connect(lambda value, spec=control: self.on_value_changed(spec, int(value)))
            row_layout.addWidget(spin)
            self.widgets[control.id] = spin

        elif control.kind == "bool":
            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(bool(current))
            checkbox.toggled.connect(lambda value, spec=control: self.on_value_changed(spec, bool(value)))
            row_layout.addWidget(checkbox)
            self.widgets[control.id] = checkbox

        elif control.kind == "enum":
            combo = QtWidgets.QComboBox()
            combo.addItems([str(option) for option in control.options])
            if str(current) in control.options:
                combo.setCurrentIndex(control.options.index(str(current)))
            combo.currentIndexChanged.connect(
                lambda idx, spec=control, options=control.options: self.on_value_changed(spec, options[int(idx)])
            )
            row_layout.addWidget(combo)
            self.widgets[control.id] = combo

        return row

    def _build_action_button(self, action: ActionSpec, state: dict[str, Any]) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(action.label)
        button.clicked.connect(lambda _checked=False, spec=action: self._invoke_action(spec, state))
        if action.shortcuts:
            button.setToolTip(f"Shortcut: {', '.join(action.shortcuts)}")
        self.widgets[action.id] = button
        return button


class ControlsHostPanel(QtWidgets.QGroupBox):
    def __init__(self, controls_panel: ControlsPanel, *, panel_id: str, title: str = "Controls", parent=None):
        super().__init__(title, parent)
        self.panel_id = panel_id
        self.controls_panel = controls_panel
        self.scroll_area = QtWidgets.QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setWidget(self.controls_panel)
        self.setMinimumHeight(0)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Ignored)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 4)
        layout.addWidget(self.scroll_area)

    def set_section_title(self, *, has_controls: bool, has_actions: bool) -> None:
        if has_controls and has_actions:
            self.setTitle("Controls & Actions")
        elif has_actions:
            self.setTitle("Actions")
        else:
            self.setTitle("Controls")


def resolve_binding(value, state: dict[str, Any]):
    if isinstance(value, StateBinding):
        return state.get(value.key)
    return value


def surface_scene_from_field(field: Field, geometry: GridGeometry | None) -> SurfaceSceneData:
    if geometry is None:
        y_dim, x_dim = field.dims[0], field.dims[1]
        y_coords = field.coord(y_dim)
        x_coords = field.coord(x_dim)
    else:
        y_dim, x_dim = geometry.dims[0], geometry.dims[1]
        y_coords = geometry.coords[y_dim]
        x_coords = geometry.coords[x_dim]

    if field.dims != (y_dim, x_dim):
        axis_map = {dim: idx for idx, dim in enumerate(field.dims)}
        z = np.transpose(field.values, axes=(axis_map[y_dim], axis_map[x_dim]))
    else:
        z = field.values

    x_grid, y_grid = np.meshgrid(np.asarray(x_coords, dtype=np.float32), np.asarray(y_coords, dtype=np.float32))
    return SurfaceSceneData(
        field_id=field.id,
        x_dim=x_dim,
        y_dim=y_dim,
        x_grid=x_grid,
        y_grid=y_grid,
        z=np.asarray(z, dtype=np.float32),
        coords={
            x_dim: np.asarray(x_coords, dtype=np.float32),
            y_dim: np.asarray(y_coords, dtype=np.float32),
        },
    )


def resolve_grid_slice_position(
    coords: dict[str, np.ndarray],
    *,
    axis_state_key: str | None,
    position_state_key: str | None,
    state: dict[str, Any],
    default_axis: str,
):
    if not axis_state_key or not position_state_key:
        return None
    axis = state.get(axis_state_key, default_axis)
    if axis not in coords:
        axis = default_axis if default_axis in coords else next(iter(coords))
    normalized = min(1.0, max(0.0, float(state.get(position_state_key, 0.0))))
    axis_coords = np.asarray(coords[axis], dtype=np.float32)
    idx = max(0, min(len(axis_coords) - 1, int(round(normalized * (len(axis_coords) - 1)))))
    return axis, idx, float(axis_coords[idx])


def overlay_from_grid_slice_operator(
    surface_scene: SurfaceSceneData,
    operator: GridSliceOperatorSpec,
    resolved_state: dict[str, Any],
):
    resolved = resolve_grid_slice_position(
        surface_scene.coords,
        axis_state_key=operator.axis_state_key,
        position_state_key=operator.position_state_key,
        state=resolved_state,
        default_axis=surface_scene.x_dim,
    )
    if resolved is None:
        return None
    axis, _idx, value = resolved
    return {
        "operator_id": operator.id,
        "axis": "x" if axis == surface_scene.x_dim else "y",
        "value": value,
        "color": resolved_state[f"{operator.id}:color"],
        "alpha": resolved_state[f"{operator.id}:alpha"],
        "fill_alpha": resolved_state[f"{operator.id}:fill_alpha"],
        "width": resolved_state[f"{operator.id}:width"],
    }


def line_from_grid_slice_operator(field: Field, operator: GridSliceOperatorSpec, state: dict[str, Any]):
    if field.values.ndim != 2:
        raise ValueError("grid slice operators require a 2D field")
    resolved = resolve_grid_slice_position(
        {dim: field.coord(dim) for dim in field.dims},
        axis_state_key=operator.axis_state_key,
        position_state_key=operator.position_state_key,
        state=state,
        default_axis=field.dims[-1],
    )
    if resolved is None:
        return None
    slice_dim, idx, slice_value = resolved
    other_dims = [dim for dim in field.dims if dim != slice_dim]
    if len(other_dims) != 1:
        raise ValueError("grid slice operators require exactly one non-sliced dimension")
    x_dim = other_dims[0]
    sliced = field.select({slice_dim: idx})
    return (
        np.asarray(sliced.coord(x_dim), dtype=np.float32),
        np.asarray(sliced.values, dtype=np.float32),
        x_dim,
        slice_dim,
        slice_value,
    )
