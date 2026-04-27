from __future__ import annotations

from collections.abc import Callable
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
from compneurovis.core.controls import (
    ActionSpec,
    BoolValueSpec,
    ChoiceValueSpec,
    ControlPresentationSpec,
    ControlSpec,
    ScalarValueSpec,
    XYValueSpec,
)
from compneurovis.core.field import Field
from compneurovis.core.geometry import GridGeometry, MorphologyGeometry
from compneurovis.core.operators import GridSliceOperatorSpec
from compneurovis.core.scene import PanelSpec
from compneurovis.core.views import LinePlotViewSpec, StateGraphViewSpec, MorphologyViewSpec, SurfaceViewSpec
from compneurovis.frontends.vispy.panel_helpers import (
    SurfaceSceneData,
    overlay_from_grid_slice_operator,
    resolve_binding,
    surface_scene_from_field,
)
from compneurovis.frontends.vispy.renderers.colormaps import _colormap_samples
from compneurovis.frontends.vispy.renderers.morphology import MorphologyRenderer
from compneurovis.frontends.vispy.renderers.surface import SurfaceRenderer


class InstrumentedSceneCanvas(scene.SceneCanvas):
    def __init__(self, *args, perf_panel_id: str | None = None, **kwargs):
        self._perf_panel_id = perf_panel_id
        self._perf_draw_count = 0
        super().__init__(*args, **kwargs)

    def on_draw(self, event) -> None:
        started = time.monotonic()
        super().on_draw(event)
        self._perf_draw_count += 1
        perf_log(
            "view_3d",
            "canvas_draw",
            panel_id=self._perf_panel_id,
            draw_count=self._perf_draw_count,
            width_px=int(self.size[0]),
            height_px=int(self.size[1]),
            duration_ms=round((time.monotonic() - started) * 1000.0, 3),
        )


class Viewport3DPanel(QtWidgets.QWidget):
    def __init__(
        self,
        *,
        host_spec: PanelSpec | None = None,
        on_entity_selected=None,
        parent=None,
    ):
        super().__init__(parent)
        self._panel_id = host_spec.id if host_spec is not None else None
        self.canvas = InstrumentedSceneCanvas(
            keys="interactive",
            bgcolor="white",
            show=False,
            perf_panel_id=self._panel_id,
        )
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
        self._active_morphology_geometry = None
        self._active_primary_renderer: str | None = None
        self._primary_renderer_clearers: dict[str, Callable[[], None]] = {}
        self._surface_scene: SurfaceSceneData | None = None
        self._surface_coord_key: tuple | None = None
        self._register_primary_renderer("morphology", self._clear_morphology_renderer)
        self._register_primary_renderer("surface", self._clear_surface_renderer)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas.native)

        self.canvas.events.mouse_press.connect(self._on_mouse_press)
        self.canvas.events.mouse_release.connect(self._on_mouse_release)

    def clear(self) -> None:
        self._activate_primary_renderer(None)
        self.canvas.native.setVisible(False)

    def refresh_morphology(
        self,
        *,
        morphology_geometry: MorphologyGeometry | None,
        morphology_view: MorphologyViewSpec | None,
        morphology_colors: np.ndarray | None,
        resolved_state: dict[str, Any],
    ) -> None:
        started = time.monotonic()
        if morphology_view is None or morphology_geometry is None:
            return

        self._activate_primary_renderer("morphology")
        self.canvas.native.setVisible(True)
        self._active_morphology_geometry = morphology_geometry
        self.canvas.bgcolor = resolved_state.get(f"{morphology_view.id}:background_color", morphology_view.background_color)
        geometry_changed = self.renderer_morph.geometry is not morphology_geometry
        set_geometry_ms = 0.0
        update_colors_ms = 0.0
        if geometry_changed:
            geometry_started = time.monotonic()
            self.renderer_morph.set_geometry(morphology_geometry)
            set_geometry_ms = round((time.monotonic() - geometry_started) * 1000.0, 3)
        if morphology_colors is not None:
            color_started = time.monotonic()
            self.renderer_morph.update_colors(
                morphology_colors,
                morphology_view.color_map,
                color_limits=resolved_state.get(f"{morphology_view.id}:color_limits", morphology_view.color_limits),
                color_norm=resolved_state.get(f"{morphology_view.id}:color_norm", morphology_view.color_norm),
            )
            update_colors_ms = round((time.monotonic() - color_started) * 1000.0, 3)
        perf_log(
            "view_3d",
            "refresh_morphology",
            panel_id=self._panel_id,
            view_id=morphology_view.id,
            geometry_changed=geometry_changed,
            segment_count=len(morphology_geometry.entity_ids),
            has_colors=morphology_colors is not None,
            set_geometry_ms=set_geometry_ms,
            update_colors_ms=update_colors_ms,
            duration_ms=round((time.monotonic() - started) * 1000.0, 3),
        )

    def refresh_surface_visual(
        self,
        *,
        surface_view: SurfaceViewSpec | None,
        surface_field: Field | None,
        grid_geometry: GridGeometry | None,
        resolved_state: dict[str, Any],
    ) -> None:
        started = time.monotonic()
        if surface_view is None or surface_field is None:
            return

        self._activate_primary_renderer("surface")
        self.canvas.native.setVisible(True)
        self.canvas.bgcolor = resolved_state[f"{surface_view.id}:background_color"]

        # Build a coord key from the grid source. Shape-based identity is sufficient -
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
            # Reuse cached x_grid/y_grid - only update z values.
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
        perf_log(
            "view_3d",
            "refresh_surface_visual",
            panel_id=self._panel_id,
            view_id=surface_view.id,
            coords_changed=coords_changed,
            field_shape=getattr(surface_field.values, "shape", None),
            duration_ms=round((time.monotonic() - started) * 1000.0, 3),
        )

    def refresh_surface_style(
        self,
        *,
        surface_view: SurfaceViewSpec | None,
        resolved_state: dict[str, Any],
    ) -> None:
        started = time.monotonic()
        if surface_view is None or self._surface_scene is None:
            return

        self._activate_primary_renderer("surface")
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
        perf_log(
            "view_3d",
            "refresh_surface_style",
            panel_id=self._panel_id,
            view_id=surface_view.id,
            duration_ms=round((time.monotonic() - started) * 1000.0, 3),
        )

    def refresh_surface_axes_geometry(
        self,
        *,
        surface_view: SurfaceViewSpec | None,
        resolved_state: dict[str, Any],
    ) -> None:
        started = time.monotonic()
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
        perf_log(
            "view_3d",
            "refresh_surface_axes_geometry",
            panel_id=self._panel_id,
            view_id=surface_view.id,
            duration_ms=round((time.monotonic() - started) * 1000.0, 3),
        )

    def refresh_surface_axes_style(
        self,
        *,
        surface_view: SurfaceViewSpec | None,
        resolved_state: dict[str, Any],
    ) -> None:
        started = time.monotonic()
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
        perf_log(
            "view_3d",
            "refresh_surface_axes_style",
            panel_id=self._panel_id,
            view_id=surface_view.id,
            duration_ms=round((time.monotonic() - started) * 1000.0, 3),
        )

    def refresh_operator_overlays(
        self,
        *,
        surface_view: SurfaceViewSpec | None,
        operators: list[GridSliceOperatorSpec],
        resolved_operator_states: dict[str, dict[str, Any]],
    ) -> None:
        started = time.monotonic()
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
        perf_log(
            "view_3d",
            "refresh_operator_overlays",
            panel_id=self._panel_id,
            view_id=surface_view.id,
            overlay_count=len(overlays),
            duration_ms=round((time.monotonic() - started) * 1000.0, 3),
        )

    def commit(self) -> None:
        started = time.monotonic()
        self.canvas.update()
        perf_log(
            "view_3d",
            "commit",
            panel_id=self._panel_id,
            active_primary_renderer=self._active_primary_renderer,
            width_px=self.width(),
            height_px=self.height(),
            duration_ms=round((time.monotonic() - started) * 1000.0, 3),
        )

    def _on_mouse_press(self, ev):
        self._mouse_start = ev.pos
        perf_log(
            "view_3d",
            "mouse_press",
            panel_id=self._panel_id,
            pos=[float(ev.pos[0]), float(ev.pos[1])],
        )

    def _on_mouse_release(self, ev):
        if self._active_morphology_geometry is None or self.on_entity_selected is None or self._mouse_start is None:
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
        perf_log(
            "view_3d",
            "mouse_release",
            panel_id=self._panel_id,
            pos=[float(ev.pos[0]), float(ev.pos[1])],
            drag_dx=float(dx),
            drag_dy=float(dy),
            picked_entity_id=entity_id,
        )
        if entity_id:
            self.on_entity_selected(entity_id)

    def _register_primary_renderer(self, key: str, clear: Callable[[], None]) -> None:
        if key in self._primary_renderer_clearers:
            raise ValueError(f"Primary 3D renderer '{key}' is already registered")
        self._primary_renderer_clearers[key] = clear

    def _activate_primary_renderer(self, key: str | None) -> None:
        if key is not None and key not in self._primary_renderer_clearers:
            raise ValueError(f"Unknown primary 3D renderer '{key}'")
        if self._active_primary_renderer == key:
            return
        if key is None:
            for clear in self._primary_renderer_clearers.values():
                clear()
        elif self._active_primary_renderer is not None:
            self._primary_renderer_clearers[self._active_primary_renderer]()
        self._active_primary_renderer = key

    def _clear_morphology_renderer(self) -> None:
        self.renderer_morph.clear()
        self._active_morphology_geometry = None

    def _clear_surface_renderer(self) -> None:
        self.renderer_surface.clear()
        self._surface_scene = None
        self._surface_coord_key = None


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


def _manual_tick_levels(xmin: float, xmax: float, major: float | None, minor: float | None):
    if major is None:
        return None
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    major_ticks = _build_tick_values(xmin, xmax, major)
    minor_ticks = _build_tick_values(xmin, xmax, minor) if minor is not None and minor > 0 else []
    major_values = {round(value, 9) for value in major_ticks}
    minor_ticks = [value for value in minor_ticks if round(value, 9) not in major_values]
    return [
        [(value, _format_tick_label(value, major)) for value in major_ticks],
        [(value, "") for value in minor_ticks],
    ]


def _build_tick_values(xmin: float, xmax: float, spacing: float | None) -> list[float]:
    if spacing is None or spacing <= 0:
        return []
    start = math.ceil((xmin - 1e-9) / spacing) * spacing
    values = []
    value = start
    while value <= xmax + 1e-9:
        values.append(round(value, 9))
        value += spacing
    return values


def _format_tick_label(value: float, spacing: float) -> str:
    if spacing >= 1 and abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    decimals = max(0, min(6, int(math.ceil(-math.log10(spacing))) if spacing < 1 else 0))
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


class LinePlotPanel(pg.PlotWidget):
    _DOWNSAMPLING_METHOD = "peak"

    def __init__(
        self,
        parent=None,
        *,
        show_internal_title: bool = True,
        perf_panel_id: str | None = None,
        perf_view_id: str | None = None,
    ):
        super().__init__(parent=parent, title="Plot" if show_internal_title else "")
        self._show_internal_title = show_internal_title
        self._perf_panel_id = perf_panel_id
        self._perf_view_id = perf_view_id
        self._resolved_title = ""
        self.setBackground("w")
        self._plot_item = self.plot([], [], pen="k")
        self._configure_data_item(self._plot_item)
        self._series_items: dict[str, pg.PlotDataItem] = {}
        self._legend_signature: tuple[str, ...] | None = None
        # Per-refresh fast-path caches. Each gates one piece of work that does
        # not depend on the data tail. Cleared via _clear_render_caches() when
        # structure changes such as view None, series clearing, or renderer swaps.
        self._cache_structural_signature: tuple[Any, ...] | None = None
        self._cache_pens: dict[str, tuple[Any, Any]] = {}
        self._cache_y_range_applied: tuple[float | None, float | None] | None = None
        self._cache_x_range_applied: tuple[float, float] | None = None
        self._cache_tick_signature: tuple[Any, ...] | str | None = None
        self._cache_background: Any = None

    def _configure_data_item(self, item: pg.PlotDataItem) -> None:
        # Let pyqtgraph clip and downsample to the visible viewport so line-plot
        # redraw cost does not grow linearly with retained history or window size.
        item.setClipToView(True)
        item.setDownsampling(auto=True, method=self._DOWNSAMPLING_METHOD)
        # This panel already strips non-finite samples before setData().
        item.setSkipFiniteCheck(True)

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
    ) -> None:
        if view is None or field is None:
            self._refresh_empty()
            return

        self._apply_background(view, state)

        sliced = self._select_field_for_view(view, field, state)
        if sliced is None:
            return

        x_dim = view.x_dim or sliced.dims[-1]
        if view.series_dim is not None:
            self._plot_item.setData([], [])
            self._refresh_series(view, sliced, x_dim, state)
            return

        self._refresh_single_trace(view, sliced, x_dim, state, source_field_id=field.id)

    def paintEvent(self, event) -> None:
        started = time.monotonic()
        super().paintEvent(event)
        duration_ms = round((time.monotonic() - started) * 1000.0, 3)
        if duration_ms >= 5.0:
            perf_log(
                "line_plot",
                "paint",
                panel_id=self._perf_panel_id,
                view_id=self._perf_view_id,
                width_px=self.width(),
                height_px=self.height(),
                duration_ms=duration_ms,
            )

    def _refresh_empty(self) -> None:
        self._clear_series()
        self._plot_item.setData([], [])
        self._set_resolved_title("")
        self._reset_view_ranges()
        self._clear_render_caches()

    def _apply_background(self, view: LinePlotViewSpec, state: dict[str, Any]) -> None:
        background = resolve_binding(view.background_color, state)
        if background is not None and background != self._cache_background:
            self.setBackground(background)
            self._cache_background = background

    def _select_field_for_view(
        self,
        view: LinePlotViewSpec,
        field: Field,
        state: dict[str, Any],
    ) -> Field | None:
        resolved_selectors = {}
        for dim, selector in view.selectors.items():
            resolved = resolve_binding(selector, state)
            if resolved is None:
                self._plot_item.setData([], [])
                return None
            filtered = self._filter_selector_for_field(field, dim, resolved)
            if filtered is None:
                self._clear_series()
                self._plot_item.setData([], [])
                return None
            resolved_selectors[dim] = filtered

        try:
            return field.select(resolved_selectors)
        except KeyError:
            self._clear_series()
            self._plot_item.setData([], [])
            return None

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

    def _refresh_single_trace(
        self,
        view: LinePlotViewSpec,
        field: Field,
        x_dim: str,
        state: dict[str, Any],
        *,
        source_field_id: str,
    ) -> None:
        self._clear_series()
        if len(field.dims) != 1 or field.dims[0] != x_dim:
            raise ValueError(f"LinePlotViewSpec '{view.id}' must resolve to a 1D field along '{x_dim}'")

        x = np.asarray(field.coord(x_dim), dtype=np.float32)
        y = np.asarray(field.values, dtype=np.float32)
        x, y = self._trim_line_data(view, x, y)
        title = view.title or source_field_id
        structural_sig = (
            "single", view.id, view.x_label or x_dim, view.x_unit,
            view.y_label, view.y_unit, title,
        )
        self._apply_single_trace_structure(
            structural_sig,
            x_label=view.x_label or x_dim,
            x_unit=view.x_unit,
            y_label=view.y_label,
            y_unit=view.y_unit,
            title=title,
        )
        self._apply_single_pen(resolve_binding(view.pen, state))
        self._plot_item.setData(x, y)
        self._apply_view_ranges(view, x)

    def _apply_single_trace_structure(
        self,
        structural_sig: tuple[Any, ...],
        *,
        x_label: str,
        x_unit: str | None,
        y_label: str,
        y_unit: str | None,
        title: str,
    ) -> None:
        if structural_sig == self._cache_structural_signature:
            return
        self.setLabel("bottom", x_label, x_unit)
        self.setLabel("left", y_label, y_unit)
        self._set_resolved_title(title)
        self._cache_structural_signature = structural_sig
        self._cache_pens.clear()

    def _apply_single_pen(self, resolved_color) -> None:
        cached_pen = self._cache_pens.get("__single__")
        if cached_pen is None or cached_pen[0] != resolved_color:
            pen = pg.mkPen(resolved_color, width=2)
            self._cache_pens["__single__"] = (resolved_color, pen)
            self._plot_item.setPen(pen)

    def _refresh_series(self, view: LinePlotViewSpec, field: Field, x_dim: str, state: dict[str, Any]) -> None:
        series_dim = view.series_dim
        if series_dim is None:
            raise ValueError("series_dim is required for multi-series refresh")
        x, values, series_labels = self._series_plot_data(view, field, x_dim, series_dim)
        self._apply_series_structure(view, field.id, x_dim, series_labels)
        self._remove_stale_series(series_labels)
        range_x = self._update_series_items(view, x, values, series_labels, state)
        self._update_series_legend(series_labels)
        self._apply_view_ranges(view, range_x)

    def _series_plot_data(
        self,
        view: LinePlotViewSpec,
        field: Field,
        x_dim: str,
        series_dim: str,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
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
        return x, values, series_labels

    def _apply_series_structure(
        self,
        view: LinePlotViewSpec,
        field_id: str,
        x_dim: str,
        series_labels: list[str],
    ) -> None:
        title = view.title or field_id
        structural_sig = (
            "series", view.id, view.x_label or x_dim, view.x_unit,
            view.y_label, view.y_unit, title, view.show_legend,
            tuple(series_labels),
        )
        if structural_sig == self._cache_structural_signature:
            return
        self.setLabel("bottom", view.x_label or x_dim, view.x_unit)
        self.setLabel("left", view.y_label, view.y_unit)
        self._set_resolved_title(title)
        self._ensure_legend(view.show_legend)
        self._cache_structural_signature = structural_sig
        self._cache_pens.clear()

    def _ensure_legend(self, enabled: bool) -> None:
        if enabled and self.plotItem.legend is None:
            self.addLegend(offset=(10, 10))
        elif not enabled and self.plotItem.legend is not None:
            self.plotItem.legend.scene().removeItem(self.plotItem.legend)
            self.plotItem.legend = None
            self._legend_signature = None

    def _remove_stale_series(self, series_labels: list[str]) -> None:
        stale = set(self._series_items.keys()) - set(series_labels)
        for label in stale:
            self.removeItem(self._series_items[label])
            del self._series_items[label]
            self._cache_pens.pop(label, None)

    def _update_series_items(
        self,
        view: LinePlotViewSpec,
        x: np.ndarray,
        values: np.ndarray,
        series_labels: list[str],
        state: dict[str, Any],
    ) -> np.ndarray:
        visible_xmin: float | None = None
        visible_xmax: float | None = None
        for idx, label in enumerate(series_labels):
            pen, pen_changed = self._series_pen(view, label, idx, state)
            item = self._series_items.get(label)
            if item is None:
                item = self.plot([], [], pen=pen)
                self._configure_data_item(item)
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

        if visible_xmin is None or visible_xmax is None:
            return np.asarray([], dtype=np.float32)
        return np.asarray([visible_xmin, visible_xmax], dtype=np.float32)

    def _series_pen(self, view: LinePlotViewSpec, label: str, idx: int, state: dict[str, Any]):
        color = self._series_color(view, label, idx)
        resolved_color = resolve_binding(color, state)
        cached = self._cache_pens.get(label)
        if cached is not None and cached[0] == resolved_color:
            return cached[1], False
        pen = pg.mkPen(resolved_color, width=2)
        self._cache_pens[label] = (resolved_color, pen)
        return pen, True

    def _series_color(self, view: LinePlotViewSpec, label: str, idx: int):
        if label in view.series_colors:
            return view.series_colors[label]
        if view.series_palette:
            return view.series_palette[idx % len(view.series_palette)]
        return view.pen

    def _update_series_legend(self, series_labels: list[str]) -> None:
        if self.plotItem.legend is not None:
            legend_signature = tuple(series_labels)
            if legend_signature != self._legend_signature:
                self.plotItem.legend.clear()
                for label in series_labels:
                    self.plotItem.legend.addItem(self._series_items[label], label)
                self._legend_signature = legend_signature
        else:
            self._legend_signature = None

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

    def _apply_view_ranges(self, view: LinePlotViewSpec, x: np.ndarray) -> None:
        self._apply_y_range(view)
        xmin, xmax = self._apply_x_range(view, x)
        self._apply_tick_spacing(view, xmin, xmax)

    def _apply_y_range(self, view: LinePlotViewSpec) -> None:
        vb = self.plotItem.getViewBox()
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

    def _apply_x_range(self, view: LinePlotViewSpec, x: np.ndarray) -> tuple[float, float]:
        vb = self.plotItem.getViewBox()
        if view.rolling_window is not None and len(x):
            data_xmin = float(np.min(x))
            data_xmax = float(np.max(x))
            xmin = max(data_xmin, data_xmax - float(view.rolling_window))
            applied = (xmin, data_xmax) if data_xmax > xmin else (xmin, xmin + max(float(view.rolling_window), 1e-6))
            if applied != self._cache_x_range_applied:
                vb.enableAutoRange(x=False)
                vb.setXRange(applied[0], applied[1], padding=0)
                self._cache_x_range_applied = applied
            return applied
        else:
            if self._cache_x_range_applied is not None:
                vb.enableAutoRange(x=True)
                vb.setLimits(xMin=None, xMax=None)
                self._cache_x_range_applied = None
            if len(x):
                return float(np.min(x)), float(np.max(x))
            return 0.0, 0.0

    def _reset_view_ranges(self) -> None:
        vb = self.plotItem.getViewBox()
        vb.enableAutoRange(x=True, y=True)
        vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
        self._reset_tick_spacing()
        self._cache_x_range_applied = None
        self._cache_y_range_applied = None
        self._cache_tick_signature = None

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
                axis.setTicks(_manual_tick_levels(xmin, xmax, major, minor))
                self._cache_tick_signature = sig
        else:
            if self._cache_tick_signature != "auto":
                self._reset_tick_spacing()
                self._cache_tick_signature = "auto"

    def _reset_tick_spacing(self) -> None:
        axis = self.plotItem.getAxis("bottom")
        axis.setTicks(None)
        axis.setTickSpacing()

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


class LinePlotHostPanel(QtWidgets.QGroupBox):
    def __init__(self, *, panel_id: str, view_id: str, title: str | None = None, parent=None):
        super().__init__(title or view_id, parent)
        self.panel_id = panel_id
        self.view_id = view_id
        self.line_plot_panel = LinePlotPanel(
            show_internal_title=False,
            perf_panel_id=panel_id,
            perf_view_id=view_id,
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 4)
        layout.addWidget(self.line_plot_panel)

    def refresh(
        self,
        view: LinePlotViewSpec | None,
        field: Field | None,
        state: dict[str, Any],
    ) -> None:
        started = time.monotonic()
        self.line_plot_panel.refresh(view, field, state)
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
                panel_width_px=self.line_plot_panel.width(),
                panel_height_px=self.line_plot_panel.height(),
            )


_LABEL_LUM_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
_MARKER_EDGE_COLOR = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _state_label_color_for_fill(rgba: np.ndarray) -> tuple[float, float, float, float]:
    luminance = float(np.dot(rgba[:3].astype(np.float32), _LABEL_LUM_WEIGHTS))
    return (1.0, 1.0, 1.0, 1.0) if luminance < 0.45 else (0.0, 0.0, 0.0, 1.0)


def _state_node_colormap_name(cmap_name: str) -> str:
    return "state-fire" if str(cmap_name).strip().lower() == "fire" else cmap_name


class StateGraphPanel(QtWidgets.QWidget):
    """VisPy canvas panel rendering a live-colored state-transition graph."""

    def __init__(self, parent=None, *, perf_panel_id: str | None = None, perf_view_id: str | None = None):
        super().__init__(parent)
        self._perf_panel_id = perf_panel_id
        self._perf_view_id = perf_view_id
        from vispy import scene as vscene
        self._vscene = vscene
        self._canvas = vscene.SceneCanvas(keys="interactive", bgcolor="white", show=False)
        self._view = self._canvas.central_widget.add_view()
        self._view.camera = vscene.cameras.PanZoomCamera(aspect=1)
        self._view.camera.set_range(x=(-0.15, 1.15), y=(-0.15, 1.15))
        self._markers = None
        self._edge_visual = None
        self._label_visual = None
        self._node_order: list[str] = []
        self._edge_order: list[str] = []
        self._node_pos: np.ndarray | None = None
        self._edge_pos: np.ndarray | None = None
        self._arrow_data: np.ndarray | None = None
        self._node_color_buf: np.ndarray | None = None
        self._label_color_buf: np.ndarray | None = None
        self._edge_color_buf: np.ndarray | None = None
        self._edge_segment_color_buf: np.ndarray | None = None
        self._field_index_cache: dict[tuple[str, tuple[str, ...], tuple[str, ...]], np.ndarray] = {}
        self._spec_sig: tuple | None = None
        lo = QtWidgets.QVBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.addWidget(self._canvas.native)

    def refresh(
        self,
        view: "StateGraphViewSpec",
        node_field: "Field | None",
        edge_field: "Field | None",
    ) -> None:
        started = time.monotonic()
        sig = (view.node_positions, view.edges, view.node_size, view.background_color,
               view.node_color_map, view.edge_color_map)
        if sig != self._spec_sig or self._markers is None:
            self._build_visuals(view)
            self._spec_sig = sig

        n = len(self._node_order)
        if n == 0:
            return

        if node_field is not None:
            nv = self._read_field_values(node_field, self._node_order, "state")
            nc = self._apply_cmap(nv, _state_node_colormap_name(view.node_color_map), view.node_color_limits)
            if self._node_color_buf is None or self._node_color_buf.shape != nc.shape:
                self._node_color_buf = np.empty_like(nc)
            self._node_color_buf[:, :] = nc
        else:
            if self._node_color_buf is None or self._node_color_buf.shape != (n, 4):
                self._node_color_buf = np.empty((n, 4), dtype=np.float32)
            self._node_color_buf[:, :] = [0.5, 0.5, 0.5, 1.0]
        self._markers.set_data(
            pos=self._node_pos, face_color=self._node_color_buf,
            size=float(view.node_size), edge_color=_MARKER_EDGE_COLOR, edge_width=1.5,
        )
        if self._label_visual is not None and self._label_color_buf is not None:
            lums = self._node_color_buf[:, :3] @ _LABEL_LUM_WEIGHTS
            self._label_color_buf[:, :3] = (lums < 0.45)[:, np.newaxis]
            self._label_visual.color = self._label_color_buf

        n_edges = len(view.edges)
        if self._edge_visual is not None and n_edges > 0:
            if edge_field is not None:
                ev = self._read_field_values(edge_field, self._edge_order, "edge")
                ec = self._apply_cmap(ev, view.edge_color_map, view.edge_color_limits)
                if self._edge_color_buf is None or self._edge_color_buf.shape != ec.shape:
                    self._edge_color_buf = np.empty_like(ec)
                self._edge_color_buf[:, :] = ec
            else:
                if self._edge_color_buf is None or self._edge_color_buf.shape != (n_edges, 4):
                    self._edge_color_buf = np.empty((n_edges, 4), dtype=np.float32)
                self._edge_color_buf[:, :] = [0.55, 0.55, 0.55, 0.85]
            if self._edge_segment_color_buf is None or self._edge_segment_color_buf.shape != (n_edges * 2, 4):
                self._edge_segment_color_buf = np.empty((n_edges * 2, 4), dtype=np.float32)
            self._edge_segment_color_buf[0::2, :] = self._edge_color_buf
            self._edge_segment_color_buf[1::2, :] = self._edge_color_buf
            self._edge_visual.set_data(color=self._edge_segment_color_buf)

        self._canvas.update()
        duration_ms = round((time.monotonic() - started) * 1000.0, 3)
        if duration_ms >= 5.0:
            perf_log(
                "state_graph", "refresh",
                panel_id=self._perf_panel_id,
                view_id=self._perf_view_id,
                duration_ms=duration_ms,
            )

    def paintEvent(self, event) -> None:
        started = time.monotonic()
        super().paintEvent(event)
        duration_ms = round((time.monotonic() - started) * 1000.0, 3)
        if duration_ms >= 5.0:
            perf_log(
                "state_graph", "paint",
                panel_id=self._perf_panel_id,
                view_id=self._perf_view_id,
                duration_ms=duration_ms,
            )

    def _build_visuals(self, view: "StateGraphViewSpec") -> None:
        vscene = self._vscene
        if self._label_visual is not None:
            self._label_visual.parent = None
            self._label_visual = None
        self._label_color_buf = None
        if self._markers is not None:
            self._markers.parent = None
            self._markers = None
        if self._edge_visual is not None:
            self._edge_visual.parent = None
            self._edge_visual = None

        self._canvas.bgcolor = view.background_color
        self._field_index_cache.clear()
        node_dict = self._node_dict(view)
        self._node_order = [name for name, x, y in view.node_positions]
        self._edge_order = [eid for src, tgt, eid in view.edges]
        n = len(self._node_order)
        if n == 0:
            self._node_pos = None
            self._node_color_buf = None
            self._edge_color_buf = None
            self._edge_segment_color_buf = None
            return

        self._node_pos = np.array(
            [[node_dict[nm][0], node_dict[nm][1]] for nm in self._node_order], dtype=np.float32
        )
        self._node_color_buf = np.full((n, 4), [0.5, 0.5, 0.5, 1.0], dtype=np.float32)
        self._markers = vscene.visuals.Markers(parent=self._view.scene)
        self._markers.set_data(
            pos=self._node_pos, face_color=self._node_color_buf,
            size=float(view.node_size), edge_color=_MARKER_EDGE_COLOR, edge_width=1.5,
        )
        self._label_color_buf = np.zeros((n, 4), dtype=np.float32)
        self._label_color_buf[:, 3] = 1.0
        self._label_visual = vscene.visuals.Text(
            text=[str(nm) for nm in self._node_order],
            pos=self._node_pos,
            color=self._label_color_buf,
            font_size=8,
            bold=True,
            anchor_x="center",
            anchor_y="center",
            parent=self._view.scene,
        )

        n_edges = len(view.edges)
        if n_edges == 0:
            return

        edge_set = {(src, tgt) for src, tgt, eid in view.edges}
        OFFSET = 0.022
        NODE_GAP = 0.035
        line_segs: list[tuple[float, float]] = []
        arrow_pts: list[list[float]] = []
        for src, tgt, eid in view.edges:
            sx, sy = node_dict[src]
            tx, ty = node_dict[tgt]
            dx, dy = tx - sx, ty - sy
            L = max(float(np.sqrt(dx * dx + dy * dy)), 1e-9)
            ux, uy = dx / L, dy / L
            px, py = -dy / L, dx / L
            if (tgt, src) in edge_set:
                ox, oy = px * OFFSET, py * OFFSET
            else:
                ox, oy = 0.0, 0.0
            x0 = sx + ox + ux * NODE_GAP
            y0 = sy + oy + uy * NODE_GAP
            x1 = tx + ox - ux * NODE_GAP
            y1 = ty + oy - uy * NODE_GAP
            line_segs.extend([(x0, y0), (x1, y1)])
            arrow_pts.append([x0, y0, x1, y1])

        self._edge_pos = np.array(line_segs, dtype=np.float32)
        self._arrow_data = np.array(arrow_pts, dtype=np.float32)
        self._edge_color_buf = np.full((n_edges, 4), [0.55, 0.55, 0.55, 0.85], dtype=np.float32)
        self._edge_segment_color_buf = np.repeat(self._edge_color_buf, 2, axis=0)
        self._edge_visual = vscene.visuals.Arrow(
            pos=self._edge_pos,
            connect="segments",
            color=self._edge_segment_color_buf,
            arrows=self._arrow_data,
            arrow_size=8,
            arrow_type="stealth",
            parent=self._view.scene,
        )

    def _node_dict(self, view: "StateGraphViewSpec") -> dict[str, tuple[float, float]]:
        return {name: (float(x), float(y)) for name, x, y in view.node_positions}

    def _read_field_values(self, field: "Field", names: list[str], dim: str) -> np.ndarray:
        coord_key = tuple(str(s) for s in field.coord(dim).tolist())
        name_key = tuple(names)
        cache_key = (dim, coord_key, name_key)
        idx = self._field_index_cache.get(cache_key)
        if idx is None:
            idx_map = {nm: i for i, nm in enumerate(coord_key)}
            idx = np.array([idx_map.get(nm, -1) for nm in name_key], dtype=np.int32)
            self._field_index_cache[cache_key] = idx

        source = np.asarray(field.values, dtype=np.float32)
        out = np.zeros(len(idx), dtype=np.float32)
        valid = idx >= 0
        if np.any(valid):
            out[valid] = source[idx[valid]]
        return out

    def _apply_cmap(self, values: np.ndarray, cmap_name: str, limits: tuple[float, float]) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        vmin, vmax = float(limits[0]), float(limits[1])
        if vmax > vmin:
            norm = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
        else:
            norm = np.zeros(len(values), dtype=np.float32)
        try:
            lut = _colormap_samples(cmap_name)
        except (KeyError, ValueError):
            lut = _colormap_samples("grays")
        idx = np.clip((norm * (len(lut) - 1)).astype(np.int32), 0, len(lut) - 1)
        return lut[idx].astype(np.float32, copy=False)


class StateGraphHostPanel(QtWidgets.QGroupBox):
    def __init__(self, *, panel_id: str, view_id: str, title: str | None = None, parent=None):
        super().__init__(title or view_id, parent)
        self.panel_id = panel_id
        self.view_id = view_id
        self.state_graph_panel = StateGraphPanel(
            perf_panel_id=panel_id, perf_view_id=view_id,
        )
        self._last_title = str(title or view_id)
        lo = QtWidgets.QVBoxLayout(self)
        lo.setContentsMargins(4, 8, 4, 4)
        lo.addWidget(self.state_graph_panel)

    def refresh(
        self,
        view: "StateGraphViewSpec | None",
        node_field: "Field | None",
        edge_field: "Field | None",
    ) -> None:
        if view is None:
            return
        title = getattr(view, "title", None) or self.view_id
        title = str(title)
        if title != self._last_title:
            self.setTitle(title)
            self._last_title = title
        self.state_graph_panel.refresh(view, node_field, edge_field)


class XYPadWidget(QtWidgets.QWidget):
    _HANDLE_RADIUS = 7
    _PAD_MARGIN = 14

    def __init__(self, control: ControlSpec, value: dict[str, float], on_changed, parent=None):
        super().__init__(parent)
        if not isinstance(control.value_spec, XYValueSpec):
            raise TypeError("XYPadWidget requires a ControlSpec with XYValueSpec")
        self._control = control
        self._spec = control.value_spec
        self._presentation = control.presentation or ControlPresentationSpec()
        self._x_norm = self._to_norm_x(float(value.get("x", self._spec.default_value()["x"])))
        self._y_norm = self._to_norm_y(float(value.get("y", self._spec.default_value()["y"])))
        self._dragging = False
        self._on_changed = on_changed
        self.setMinimumSize(160, 175)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)

    def _to_norm_x(self, value: float) -> float:
        x_min, x_max = self._spec.x_range
        span = x_max - x_min
        return max(0.0, min(1.0, (value - x_min) / span)) if span else 0.5

    def _to_norm_y(self, value: float) -> float:
        y_min, y_max = self._spec.y_range
        span = y_max - y_min
        return max(0.0, min(1.0, (value - y_min) / span)) if span else 0.5

    def _pad_rect(self) -> tuple[int, int, int, int]:
        m = self._PAD_MARGIN
        label_reserve = 18
        w = self.width() - 2 * m
        h = self.height() - 2 * m - label_reserve
        side = max(1, min(w, h))
        x0 = m + (w - side) // 2
        y0 = m
        return x0, y0, side, side

    def _norm_to_pixel(self, nx: float, ny: float) -> tuple[float, float]:
        x0, y0, w, h = self._pad_rect()
        return x0 + nx * w, y0 + (1.0 - ny) * h

    def _pixel_to_norm(self, px: float, py: float) -> tuple[float, float]:
        x0, y0, w, h = self._pad_rect()
        nx = max(0.0, min(1.0, (px - x0) / w)) if w else 0.5
        ny = max(0.0, min(1.0, 1.0 - (py - y0) / h)) if h else 0.5
        return nx, ny

    def _norm_to_values(self, nx: float, ny: float) -> dict[str, float]:
        x_min, x_max = self._spec.x_range
        y_min, y_max = self._spec.y_range
        return {
            "x": float(x_min + nx * (x_max - x_min)),
            "y": float(y_min + ny * (y_max - y_min)),
        }

    def paintEvent(self, event) -> None:
        from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QPainterPath
        from PyQt6.QtCore import QRectF, QPointF

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        x0, y0, w, h = self._pad_rect()
        pad_rect = QRectF(x0, y0, w, h)
        bg = QColor(40, 40, 45)
        border = QColor(80, 80, 92)
        grid_color = QColor(60, 60, 70)

        if self._presentation.shape == "circle":
            painter.setBrush(QBrush(bg))
            painter.setPen(QPen(border, 1.5))
            painter.drawEllipse(pad_rect)
            clip = QPainterPath()
            clip.addEllipse(pad_rect)
            painter.setClipPath(clip)
        else:
            painter.setBrush(QBrush(bg))
            painter.setPen(QPen(border, 1.5))
            painter.drawRoundedRect(pad_rect, 4.0, 4.0)

        cx, cy = self._norm_to_pixel(0.5, 0.5)
        painter.setPen(QPen(grid_color, 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(x0, cy), QPointF(x0 + w, cy))
        painter.drawLine(QPointF(cx, y0), QPointF(cx, y0 + h))

        painter.setClipping(False)

        hx, hy = self._norm_to_pixel(self._x_norm, self._y_norm)
        r = self._HANDLE_RADIUS

        painter.setBrush(QBrush(QColor(100, 180, 255, 55)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(hx - r - 4, hy - r - 4, (r + 4) * 2, (r + 4) * 2))

        painter.setBrush(QBrush(QColor(100, 180, 255)))
        painter.setPen(QPen(QColor(210, 235, 255), 1.5))
        painter.drawEllipse(QRectF(hx - r, hy - r, r * 2, r * 2))

        value = self._norm_to_values(self._x_norm, self._y_norm)
        painter.setPen(QPen(QColor(155, 155, 175)))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        label = f"{self._spec.x_label}: {value['x']:.3g}   {self._spec.y_label}: {value['y']:.3g}"
        painter.drawText(int(x0), int(y0 + h + self._PAD_MARGIN), label)

        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._update_from_pos(event.position().x(), event.position().y())

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self._update_from_pos(event.position().x(), event.position().y())

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False

    def _update_from_pos(self, px: float, py: float) -> None:
        nx, ny = self._pixel_to_norm(px, py)
        self._x_norm = nx
        self._y_norm = ny
        self._on_changed(self._norm_to_values(nx, ny))
        self.update()

    def set_values(self, value: dict[str, float]) -> None:
        self._x_norm = self._to_norm_x(float(value.get("x", self._spec.default_value()["x"])))
        self._y_norm = self._to_norm_y(float(value.get("y", self._spec.default_value()["y"])))
        self.update()


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

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rebuild_grid(force=False)

    def _desired_column_count(self) -> int:
        scalar_count = sum(1 for c in self._controls if not isinstance(c.value_spec, XYValueSpec))
        item_count = scalar_count + len(self._actions)
        if item_count < self._MULTI_COLUMN_MIN_ITEMS:
            return 1
        if self.width() < self._MULTI_COLUMN_MIN_WIDTH:
            return 1
        return 2

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
        current_col = 0
        for control in self._controls:
            if isinstance(control.value_spec, XYValueSpec):
                if current_col > 0:
                    row_index += 1
                    current_col = 0
                self._grid.addWidget(self._build_xy_pad_row(control, self._state), row_index, 0, 1, column_count)
                row_index += 1
            else:
                self._grid.addWidget(self._build_control_row(control, self._state), row_index, current_col)
                current_col += 1
                if current_col >= column_count:
                    current_col = 0
                    row_index += 1
        if current_col > 0:
            row_index += 1

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

    def _clear_grid(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _build_control_row(self, control: ControlSpec, state: dict[str, Any]) -> QtWidgets.QWidget:
        row, row_layout = self._control_row_shell(control)
        current = self._control_current_value(control, state)
        value_spec = control.value_spec
        presentation = control.presentation or ControlPresentationSpec()

        if isinstance(value_spec, ScalarValueSpec) and value_spec.value_type == "float":
            self._add_float_control(row_layout, control, value_spec, presentation, current)
        elif isinstance(value_spec, ScalarValueSpec) and value_spec.value_type == "int":
            self._add_int_control(row_layout, control, value_spec, presentation, current)
        elif isinstance(value_spec, BoolValueSpec):
            self._add_bool_control(row_layout, control, presentation, current)
        elif isinstance(value_spec, ChoiceValueSpec):
            self._add_choice_control(row_layout, control, value_spec, presentation, current)
        else:
            raise ValueError(f"Unsupported value spec for control '{control.id}'")

        return row

    def _control_row_shell(self, control: ControlSpec) -> tuple[QtWidgets.QWidget, QtWidgets.QHBoxLayout]:
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(QtWidgets.QLabel(control.label))
        return row, row_layout

    def _control_current_value(self, control: ControlSpec, state: dict[str, Any]):
        return state.get(control.resolved_state_key(), control.default_value())

    def _validate_control_kind(self, *, kind: str | None, default: str, expected: str, control: ControlSpec, label: str):
        resolved_kind = kind or default
        if resolved_kind != expected:
            raise ValueError(f"Unsupported presentation kind '{resolved_kind}' for {label} control '{control.id}'")
        return resolved_kind

    def _add_float_control(
        self,
        row_layout: QtWidgets.QHBoxLayout,
        control: ControlSpec,
        value_spec: ScalarValueSpec,
        presentation: ControlPresentationSpec,
        current: Any,
    ) -> None:
        self._validate_control_kind(
            kind=presentation.kind,
            default="slider",
            expected="slider",
            control=control,
            label="scalar float",
        )
        slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        steps = int(presentation.steps or 100)
        slider.setRange(0, steps)
        min_value = float(value_spec.min if value_spec.min is not None else 0.0)
        max_value = float(value_spec.max if value_spec.max is not None else 1.0)
        value_label = QtWidgets.QLabel("")

        def on_change(raw: int, *, spec=control, label=value_label) -> None:
            scale = (spec.presentation or ControlPresentationSpec()).scale
            value = self._slider_raw_to_value(
                raw,
                min_value=min_value,
                max_value=max_value,
                steps=steps,
                scale=scale,
            )
            label.setText(f"{value:.3g}")
            self.on_value_changed(spec, value)

        raw_value = self._slider_value_to_raw(
            current,
            min_value=min_value,
            max_value=max_value,
            steps=steps,
            scale=presentation.scale,
        )
        slider.setValue(max(0, min(steps, raw_value)))
        slider.valueChanged.connect(on_change)
        initial_value = self._slider_raw_to_value(
            slider.value(),
            min_value=min_value,
            max_value=max_value,
            steps=steps,
            scale=presentation.scale,
        )
        value_label.setText(f"{initial_value:.3g}")
        row_layout.addWidget(slider, 1)
        row_layout.addWidget(value_label)
        self.widgets[control.id] = slider

    def _add_int_control(
        self,
        row_layout: QtWidgets.QHBoxLayout,
        control: ControlSpec,
        value_spec: ScalarValueSpec,
        presentation: ControlPresentationSpec,
        current: Any,
    ) -> None:
        self._validate_control_kind(
            kind=presentation.kind,
            default="spinbox",
            expected="spinbox",
            control=control,
            label="scalar int",
        )
        spin = QtWidgets.QSpinBox()
        spin.setRange(
            int(value_spec.min if value_spec.min is not None else 0),
            int(value_spec.max if value_spec.max is not None else 100),
        )
        spin.setValue(int(current))
        spin.valueChanged.connect(lambda value, spec=control: self.on_value_changed(spec, int(value)))
        row_layout.addWidget(spin)
        self.widgets[control.id] = spin

    def _add_bool_control(
        self,
        row_layout: QtWidgets.QHBoxLayout,
        control: ControlSpec,
        presentation: ControlPresentationSpec,
        current: Any,
    ) -> None:
        self._validate_control_kind(
            kind=presentation.kind,
            default="checkbox",
            expected="checkbox",
            control=control,
            label="bool",
        )
        checkbox = QtWidgets.QCheckBox()
        checkbox.setChecked(bool(current))
        checkbox.toggled.connect(lambda value, spec=control: self.on_value_changed(spec, bool(value)))
        row_layout.addWidget(checkbox)
        self.widgets[control.id] = checkbox

    def _add_choice_control(
        self,
        row_layout: QtWidgets.QHBoxLayout,
        control: ControlSpec,
        value_spec: ChoiceValueSpec,
        presentation: ControlPresentationSpec,
        current: Any,
    ) -> None:
        self._validate_control_kind(
            kind=presentation.kind,
            default="dropdown",
            expected="dropdown",
            control=control,
            label="choice",
        )
        combo = QtWidgets.QComboBox()
        combo.addItems([str(option) for option in value_spec.options])
        if str(current) in value_spec.options:
            combo.setCurrentIndex(value_spec.options.index(str(current)))
        combo.currentIndexChanged.connect(
            lambda idx, spec=control, options=value_spec.options: self.on_value_changed(spec, options[int(idx)])
        )
        row_layout.addWidget(combo)
        self.widgets[control.id] = combo

    def _build_xy_pad_row(self, control: ControlSpec, state: dict[str, Any]) -> QtWidgets.QWidget:
        wrapper = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)
        if control.label:
            layout.addWidget(QtWidgets.QLabel(control.label))
        if not isinstance(control.value_spec, XYValueSpec):
            raise ValueError(f"Control '{control.id}' is not an XY control")
        presentation = control.presentation or ControlPresentationSpec()
        kind = presentation.kind or "xy_pad"
        if kind != "xy_pad":
            raise ValueError(f"Unsupported presentation kind '{kind}' for XY control '{control.id}'")
        current = state.get(control.resolved_state_key(), control.default_value())
        if not isinstance(current, dict):
            current = control.default_value()

        def on_xy_changed(value: dict[str, float], spec=control) -> None:
            self.on_value_changed(spec, value)

        pad = XYPadWidget(control, current, on_xy_changed)
        layout.addWidget(pad)
        self.widgets[control.id] = pad
        return wrapper

    def _build_action_button(self, action: ActionSpec, state: dict[str, Any]) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(action.label)
        button.clicked.connect(lambda _checked=False, spec=action: self._invoke_action(spec, state))
        if action.shortcuts:
            button.setToolTip(f"Shortcut: {', '.join(action.shortcuts)}")
        self.widgets[action.id] = button
        return button

    @staticmethod
    def _slider_raw_to_value(raw: int, *, min_value: float, max_value: float, steps: int, scale: str) -> float:
        frac = raw / max(1, steps)
        if scale == "log" and min_value > 0 and max_value > min_value:
            return float(min_value * ((max_value / min_value) ** frac))
        return float(min_value + (max_value - min_value) * frac)

    @staticmethod
    def _slider_value_to_raw(value: Any, *, min_value: float, max_value: float, steps: int, scale: str) -> int:
        try:
            numeric = float(value)
        except Exception:
            return 0
        if max_value <= min_value:
            return 0
        if scale == "log" and min_value > 0 and max_value > min_value:
            if numeric <= 0:
                return 0
            frac = math.log(numeric / min_value) / math.log(max_value / min_value)
        else:
            frac = (numeric - min_value) / (max_value - min_value)
        return int(round(min(max(frac, 0.0), 1.0) * steps))

    def _invoke_action(self, action: ActionSpec, state: dict[str, Any]) -> None:
        if self.on_action_invoked is None:
            return
        payload = {
            key: resolve_binding(value, state)
            for key, value in action.payload.items()
        }
        self.on_action_invoked(action, payload)


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
