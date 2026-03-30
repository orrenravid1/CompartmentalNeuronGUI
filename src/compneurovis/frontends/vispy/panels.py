from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
from vispy import scene
from vispy.scene.cameras import TurntableCamera

from compneurovis.core.controls import ActionSpec, ControlSpec
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
        self._series_items: dict[str, pg.PlotDataItem] = {}
        self._legend_signature: tuple[str, ...] | None = None

    def refresh(self, view: LinePlotViewSpec | None, field: Field | None, state: dict[str, Any], geometry_lookup: dict[str, MorphologyGeometry]) -> None:
        if view is None or field is None:
            self._clear_series()
            self._plot_item.setData([], [])
            self.setTitle("")
            self._reset_view_ranges()
            return

        background = resolve_binding(view.background_color, state)
        if background is not None:
            self.setBackground(background)

        if view.orthogonal_slice_state_key:
            self._clear_series()
            line = line_from_orthogonal_slice(field, view, state)
            if line is None:
                self._plot_item.setData([], [])
                return
            x, y, x_dim, slice_dim, slice_value = line
            x, y = self._trim_line_data(view, x, y)
            self.setLabel("bottom", x_dim, view.x_unit)
            self.setLabel("left", view.y_label, view.y_unit)
            title = view.title or field.id
            self.setTitle(f"{title} at {slice_dim} = {slice_value:.3f}")
            self._plot_item.setPen(pg.mkPen(resolve_binding(view.pen, state), width=2))
            self._plot_item.setData(x, y)
            self._apply_view_ranges(view, x)
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
        self._apply_view_ranges(view, x)

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

    def _refresh_series(self, view: LinePlotViewSpec, field: Field, x_dim: str, state: dict[str, Any]) -> None:
        series_dim = view.series_dim
        if series_dim is None:
            raise ValueError("series_dim is required for multi-series refresh")
        if set(field.dims) != {series_dim, x_dim} or field.values.ndim != 2:
            raise ValueError(
                f"LinePlotViewSpec '{view.id}' with series_dim='{series_dim}' must resolve to a 2D field over ({series_dim}, {x_dim})"
            )

        axis_map = {dim: idx for idx, dim in enumerate(field.dims)}
        values = np.asarray(field.values, dtype=np.float32)
        if field.dims != (series_dim, x_dim):
            values = np.transpose(values, axes=(axis_map[series_dim], axis_map[x_dim]))

        x = np.asarray(field.coord(x_dim), dtype=np.float32)
        series_labels = [str(label) for label in field.coord(series_dim)]
        x, values = self._trim_series_data(view, x, values)
        self.setLabel("bottom", view.x_label or x_dim, view.x_unit)
        self.setLabel("left", view.y_label, view.y_unit)
        self.setTitle(view.title or field.id)
        self._ensure_legend(view.show_legend)

        stale = set(self._series_items.keys()) - set(series_labels)
        for label in stale:
            self.removeItem(self._series_items[label])
            del self._series_items[label]

        for idx, label in enumerate(series_labels):
            if label in view.series_colors:
                color = view.series_colors[label]
            elif view.series_palette:
                color = view.series_palette[idx % len(view.series_palette)]
            else:
                color = view.pen
            pen = pg.mkPen(resolve_binding(color, state), width=2)
            item = self._series_items.get(label)
            if item is None:
                item = self.plot([], [], pen=pen)
                self._series_items[label] = item
            else:
                item.setPen(pen)
            item.setData(x, values[idx])
        if self.plotItem.legend is not None:
            legend_signature = tuple(series_labels)
            if legend_signature != self._legend_signature:
                self.plotItem.legend.clear()
                for label in series_labels:
                    self.plotItem.legend.addItem(self._series_items[label], label)
                self._legend_signature = legend_signature
        else:
            self._legend_signature = None
        self._apply_view_ranges(view, x)

    def _reset_view_ranges(self) -> None:
        vb = self.plotItem.getViewBox()
        vb.enableAutoRange(x=True, y=True)
        vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
        self._reset_tick_spacing()

    def _apply_view_ranges(self, view: LinePlotViewSpec, x: np.ndarray) -> None:
        vb = self.plotItem.getViewBox()
        xmin = 0.0
        xmax = 0.0
        data_xmin = 0.0
        data_xmax = 0.0

        if view.y_min is not None or view.y_max is not None:
            y_min = view.y_min
            y_max = view.y_max
            vb.enableAutoRange(y=False)
            vb.setLimits(yMin=y_min, yMax=y_max)
            if y_min is not None and y_max is not None:
                vb.setYRange(float(y_min), float(y_max), padding=0)
        else:
            vb.enableAutoRange(y=True)
            vb.setLimits(yMin=None, yMax=None)

        if len(x):
            data_xmin = float(np.min(x))
            data_xmax = float(np.max(x))

        if view.rolling_window is not None and len(x):
            xmax = data_xmax
            xmin = max(data_xmin, xmax - float(view.rolling_window))
            vb.enableAutoRange(x=False)
            vb.setXRange(xmin, xmax, padding=0)
        else:
            if len(x):
                xmin = data_xmin
                xmax = data_xmax
            vb.enableAutoRange(x=True)
            vb.setLimits(xMin=None, xMax=None)

        self._apply_tick_spacing(view, xmin, xmax)

    def _trim_line_data(self, view: LinePlotViewSpec, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not view.trim_to_rolling_window or view.rolling_window is None or len(x) == 0:
            return x, y
        mask = self._rolling_window_mask(x, float(view.rolling_window))
        return x[mask], y[mask]

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
            axis.setTicks(self._manual_tick_levels(xmin, xmax, major, minor))
        else:
            self._reset_tick_spacing()

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


class ControlsPanel(QtWidgets.QWidget):
    def __init__(self, on_value_changed, on_action_invoked=None, parent=None):
        super().__init__(parent)
        self.on_value_changed = on_value_changed
        self.on_action_invoked = on_action_invoked
        self.widgets: dict[str, QtWidgets.QWidget] = {}
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.layout.setSpacing(6)

    def set_controls(self, controls: list[ControlSpec], actions: list[ActionSpec], state: dict[str, Any]) -> None:
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

        if actions:
            if controls:
                divider = QtWidgets.QFrame()
                divider.setFrameShape(QtWidgets.QFrame.Shape.HLine)
                divider.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
                self.layout.addWidget(divider)

            for action in actions:
                button = QtWidgets.QPushButton(action.label)
                button.clicked.connect(lambda _checked=False, spec=action: self._invoke_action(spec, state))
                if action.shortcuts:
                    button.setToolTip(f"Shortcut: {', '.join(action.shortcuts)}")
                self.layout.addWidget(button)
                self.widgets[action.id] = button

        self.layout.addStretch(1)

    def _invoke_action(self, action: ActionSpec, state: dict[str, Any]) -> None:
        if self.on_action_invoked is None:
            return
        payload = {
            key: resolve_binding(value, state)
            for key, value in action.payload.items()
        }
        self.on_action_invoked(action, payload)


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
