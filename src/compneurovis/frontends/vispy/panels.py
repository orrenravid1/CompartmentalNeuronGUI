from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
from vispy import scene
from vispy.scene.cameras import TurntableCamera

from compneurovis.core.controls import ControlSpec
from compneurovis.core.field import Field
from compneurovis.core.geometry import GridGeometry, MorphologyGeometry
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
    def __init__(self, on_entity_selected=None, parent=None):
        super().__init__(parent)
        self.canvas = scene.SceneCanvas(keys="interactive", bgcolor="white", show=False)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = TurntableCamera(
            fov=60,
            distance=200,
            elevation=30,
            azimuth=30,
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
            self.renderer_morph.update_colors(morphology_colors, morphology_view.color_map)

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
        self._surface_scene = surface_scene_from_field(surface_field, grid_geometry)
        self.renderer_surface.update_surface(
            self._surface_scene.x_grid,
            self._surface_scene.y_grid,
            self._surface_scene.z,
            cmap=resolved_state[f"{surface_view.id}:cmap"],
            clim=resolved_state[f"{surface_view.id}:clim"],
            colors=None,
            color_by=resolved_state[f"{surface_view.id}:color_by"],
            surface_alpha=resolved_state[f"{surface_view.id}:surface_alpha"],
        )

    def refresh_surface_axes(
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
        self.renderer_surface.axes.set_axes(
            render_axes=resolved_state[f"{surface_view.id}:render_axes"],
            axes_in_middle=resolved_state[f"{surface_view.id}:axes_in_middle"],
            tick_count=resolved_state[f"{surface_view.id}:tick_count"],
            tick_length_scale=resolved_state[f"{surface_view.id}:tick_length_scale"],
            tick_label_size=resolved_state[f"{surface_view.id}:tick_label_size"],
            axis_label_size=resolved_state[f"{surface_view.id}:axis_label_size"],
            axis_color=resolved_state[f"{surface_view.id}:axis_color"],
            text_color=resolved_state[f"{surface_view.id}:text_color"],
            axis_alpha=resolved_state[f"{surface_view.id}:axis_alpha"],
            axis_labels=axis_labels,
            x=self._surface_scene.x_grid,
            y=self._surface_scene.y_grid,
            z=self._surface_scene.z,
        )

    def refresh_surface_slice(
        self,
        *,
        surface_view: SurfaceViewSpec | None,
        resolved_state: dict[str, Any],
    ) -> None:
        if surface_view is None or self._surface_scene is None:
            self.renderer_surface.slice_overlay.clear()
            return

        axis_key = surface_view.slice_axis_state_key
        pos_key = surface_view.slice_position_state_key
        if not axis_key or not pos_key:
            self.renderer_surface.slice_overlay.clear()
            return

        axis = resolved_state.get(axis_key, self._surface_scene.x_dim)
        if axis not in self._surface_scene.coords:
            axis = self._surface_scene.x_dim
        normalized = float(resolved_state.get(pos_key, 0.0))
        coords = self._surface_scene.coords[axis]
        idx = max(0, min(len(coords) - 1, int(round(normalized * (len(coords) - 1)))))
        value = float(coords[idx])
        self.renderer_surface.slice_overlay.set_slice(
            axis="x" if axis == self._surface_scene.x_dim else "y",
            value=value,
            color=resolved_state[f"{surface_view.id}:slice_color"],
            alpha=resolved_state[f"{surface_view.id}:slice_alpha"],
            width=resolved_state[f"{surface_view.id}:slice_width"],
            x=self._surface_scene.x_grid,
            y=self._surface_scene.y_grid,
            z=self._surface_scene.z,
        )

    def commit(self) -> None:
        self.canvas.update()


class LinePlotPanel(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent, title="Plot")
        self.setBackground("w")
        self._plot_item = self.plot([], [], pen="k")

    def refresh(self, view: LinePlotViewSpec | None, field: Field | None, state: dict[str, Any], geometry_lookup: dict[str, MorphologyGeometry]) -> None:
        if view is None or field is None:
            self._plot_item.setData([], [])
            self.setTitle("")
            return

        background = resolve_binding(view.background_color, state)
        if background is not None:
            self.setBackground(background)

        if view.orthogonal_slice_state_key:
            line = line_from_orthogonal_slice(field, view, state)
            if line is None:
                self._plot_item.setData([], [])
                return
            x, y, x_dim, slice_dim, slice_value = line
            self.setLabel("bottom", x_dim, view.x_unit)
            self.setLabel("left", view.y_label, view.y_unit)
            title = view.title or field.id
            self.setTitle(f"{title} at {slice_dim} = {slice_value:.3f}")
            self._plot_item.setPen(pg.mkPen(resolve_binding(view.pen, state), width=2))
            self._plot_item.setData(x, y)
            return

        resolved_selectors = {}
        for dim, selector in view.selectors.items():
            resolved = resolve_binding(selector, state)
            if resolved is None:
                self._plot_item.setData([], [])
                return
            resolved_selectors[dim] = resolved

        sliced = field.select(resolved_selectors)
        x_dim = view.x_dim or sliced.dims[-1]
        if len(sliced.dims) != 1 or sliced.dims[0] != x_dim:
            raise ValueError(f"LinePlotViewSpec '{view.id}' must resolve to a 1D field along '{x_dim}'")

        x = np.asarray(sliced.coord(x_dim), dtype=np.float32)
        y = np.asarray(sliced.values, dtype=np.float32)
        self.setLabel("bottom", view.x_label or x_dim, view.x_unit)
        self.setLabel("left", view.y_label, view.y_unit)
        title = view.title or field.id
        entity_id = state.get("selected_entity_id")
        if entity_id:
            for geometry in geometry_lookup.values():
                if entity_id in geometry.entity_ids:
                    title = f"{title}: {geometry.label_for(entity_id)}"
                    break
        self.setTitle(title)
        self._plot_item.setPen(pg.mkPen(resolve_binding(view.pen, state), width=2))
        self._plot_item.setData(x, y)


class ControlsPanel(QtWidgets.QWidget):
    def __init__(self, on_value_changed, parent=None):
        super().__init__(parent)
        self.on_value_changed = on_value_changed
        self.widgets: dict[str, QtWidgets.QWidget] = {}
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.layout.setSpacing(6)

    def set_controls(self, controls: list[ControlSpec], state: dict[str, Any]) -> None:
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.widgets.clear()

        for control in controls:
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

            self.layout.addWidget(row)

        self.layout.addStretch(1)


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


def line_from_orthogonal_slice(field: Field, view: LinePlotViewSpec, state: dict[str, Any]):
    if field.values.ndim != 2:
        raise ValueError("orthogonal slice line plots require a 2D field")
    slice_dim = state.get(view.orthogonal_slice_state_key)
    if slice_dim not in field.dims:
        slice_dim = field.dims[0]
    other_dims = [dim for dim in field.dims if dim != slice_dim]
    if len(other_dims) != 1:
        raise ValueError("orthogonal slice line plots require exactly one non-sliced dimension")
    x_dim = other_dims[0]
    pos = float(state.get(view.orthogonal_position_state_key, 0.0))
    coords = field.coord(slice_dim)
    idx = max(0, min(len(coords) - 1, int(round(pos * (len(coords) - 1)))))
    slice_value = float(coords[idx])
    sliced = field.select({slice_dim: idx})
    return (
        np.asarray(sliced.coord(x_dim), dtype=np.float32),
        np.asarray(sliced.values, dtype=np.float32),
        x_dim,
        slice_dim,
        slice_value,
    )
