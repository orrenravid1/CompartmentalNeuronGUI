from __future__ import annotations

from functools import lru_cache
import time

import numpy as np
from vispy import scene
from vispy.color import Color

from compneurovis.core.geometry import MorphologyGeometry
from compneurovis.vispyutils.cappedcylindercollection import CappedCylinderCollection


_RAMP_LOW_COLOR_MIX = 0.2


def _single_color_ramp(color, n: int = 256) -> np.ndarray:
    high = np.asarray(Color(color).rgba, dtype=np.float32)
    low = high.copy()
    low[:3] = 1.0 - _RAMP_LOW_COLOR_MIX * (1.0 - high[:3])
    alpha = np.linspace(low[3], high[3], n, dtype=np.float32)[:, None]
    rgb = np.linspace(low[:3], high[:3], n, dtype=np.float32)
    return np.concatenate([rgb, alpha], axis=1)


def _two_color_ramp(low_color, high_color, n: int = 256) -> np.ndarray:
    return _multi_color_ramp((low_color, high_color), n=n)


def _multi_color_ramp(colors, n: int = 256) -> np.ndarray:
    rgba = np.asarray([Color(color).rgba for color in colors], dtype=np.float32)
    if rgba.shape[0] == 0:
        raise ValueError("At least one color is required for a ramp colormap")
    if rgba.shape[0] == 1:
        return _single_color_ramp(colors[0], n=n)
    ramp_positions = np.linspace(0.0, 1.0, rgba.shape[0], dtype=np.float32)
    sample_positions = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.stack(
        [
            np.interp(sample_positions, ramp_positions, rgba[:, channel]).astype(np.float32)
            for channel in range(4)
        ],
        axis=1,
    )


def _sample_matplotlib_colormap(name: str, n: int = 256) -> np.ndarray:
    try:
        from matplotlib import colormaps
    except ImportError as exc:
        raise ValueError(
            f"Matplotlib colormap '{name}' requested via 'mpl:' but matplotlib is not installed"
        ) from exc
    try:
        cmap = colormaps[name]
    except KeyError as exc:
        raise ValueError(f"Unknown matplotlib colormap '{name}'") from exc
    samples = np.asarray(cmap(np.linspace(0.0, 1.0, n, dtype=np.float32)), dtype=np.float32)
    return samples


def _sample_matplotlib_ramp(colors, n: int = 256) -> np.ndarray:
    try:
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        return _multi_color_ramp(colors, n=n)
    cmap = LinearSegmentedColormap.from_list("compneurovis-custom-ramp", list(colors))
    samples = np.asarray(cmap(np.linspace(0.0, 1.0, n, dtype=np.float32)), dtype=np.float32)
    return samples


@lru_cache(maxsize=64)
def _cached_colormap_samples(name: str, n: int = 256) -> np.ndarray:
    raw_name = str(name).strip()
    normalized = raw_name.lower()
    if normalized.startswith("mpl-ramp:"):
        colors = tuple(color.strip() or "#000000" for color in raw_name.split(":")[1:] if color.strip())
        if len(colors) < 2:
            raise ValueError("Matplotlib ramp colormaps require at least two colors after 'mpl-ramp:'")
        return _sample_matplotlib_ramp(colors, n=n)
    if normalized.startswith("mpl:"):
        cmap_name = raw_name.split(":", 1)[1].strip()
        if not cmap_name:
            raise ValueError("Matplotlib colormaps require a name after 'mpl:'")
        return _sample_matplotlib_colormap(cmap_name, n=n)
    if normalized.startswith("ramp:"):
        colors = tuple(color.strip() or "#000000" for color in raw_name.split(":")[1:] if color.strip())
        if not colors:
            raise ValueError("Ramp colormaps require at least one color after 'ramp:'")
        if len(colors) == 1:
            return _single_color_ramp(colors[0], n=n)
        return _multi_color_ramp(colors, n=n)
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    if normalized == "grayscale":
        rgb = np.stack([x, x, x], axis=1)
    elif normalized in {"markov-fire", "white-fire"}:
        return _multi_color_ramp(("#ffffff", "#ffe45c", "#f18f01", "#8f0500"), n=n)
    elif normalized == "fire":
        rgb = np.stack(
            [
                np.clip(1.5 * x, 0.0, 1.0),
                np.clip(2.0 * x - 0.4, 0.0, 1.0),
                np.clip(4.0 * x - 3.0, 0.0, 1.0),
            ],
            axis=1,
        )
    else:
        rgb = np.empty((n, 3), dtype=np.float32)
        left = x <= 0.5
        right = ~left
        rgb[left, 0] = 2.0 * x[left]
        rgb[left, 1] = 2.0 * x[left]
        rgb[left, 2] = 1.0
        rgb[right, 0] = 1.0
        rgb[right, 1] = 2.0 * (1.0 - x[right])
        rgb[right, 2] = 2.0 * (1.0 - x[right])
    alpha = np.ones((n, 1), dtype=np.float32)
    return np.concatenate([rgb, alpha], axis=1)


def _colormap_samples(name: str, n: int = 256) -> np.ndarray:
    return _cached_colormap_samples(str(name), int(n))


class SurfaceAxesOverlay:
    _TICK_ANCHORS = {
        "x": ("center", "top"),
        "y": ("right", "center"),
        "z": ("left", "center"),
    }

    _AXIS_LABEL_ANCHORS = {
        "x": ("center", "top"),
        "y": ("right", "center"),
        "z": ("left", "center"),
    }

    def __init__(self, view):
        self.view = view
        self._axis_lines = None
        self._tick_lines = None
        self._tick_labels: dict[str, scene.visuals.Text] = {}
        self._axis_labels: dict[str, scene.visuals.Text] = {}

    def clear(self) -> None:
        visuals = [
            self._axis_lines,
            self._tick_lines,
            *self._tick_labels.values(),
            *self._axis_labels.values(),
        ]
        for vis in visuals:
            if vis is None:
                continue
            vis.parent = None
        self._axis_lines = None
        self._tick_lines = None
        self._tick_labels.clear()
        self._axis_labels.clear()

    def _ensure_line_visual(self, *, attr_name: str, width: float):
        visual = getattr(self, attr_name)
        if visual is not None:
            return visual
        visual = scene.visuals.Line(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=(0.0, 0.0, 0.0, 1.0),
            width=width,
            connect="segments",
            method="gl",
            parent=self.view.scene,
        )
        visual.set_gl_state(depth_test=False, blend=True)
        visual.order = 1000
        setattr(self, attr_name, visual)
        return visual

    def _ensure_tick_label_visual(self, axis_name: str):
        visual = self._tick_labels.get(axis_name)
        if visual is not None:
            return visual
        anchor_x, anchor_y = self._TICK_ANCHORS[axis_name]
        visual = scene.visuals.Text(
            text=[],
            pos=np.zeros((1, 3), dtype=np.float32),
            color="black",
            font_size=10,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
            parent=self.view.scene,
        )
        visual.set_gl_state(depth_test=False, blend=True)
        visual.order = 1000
        self._tick_labels[axis_name] = visual
        return visual

    def _ensure_axis_label_visual(self, axis_name: str):
        visual = self._axis_labels.get(axis_name)
        if visual is not None:
            return visual
        anchor_x, anchor_y = self._AXIS_LABEL_ANCHORS[axis_name]
        visual = scene.visuals.Text(
            text="",
            pos=np.zeros((1, 3), dtype=np.float32),
            color="black",
            font_size=12,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
            parent=self.view.scene,
        )
        visual.set_gl_state(depth_test=False, blend=True)
        visual.order = 1000
        self._axis_labels[axis_name] = visual
        return visual

    def _set_visible(self, visible: bool) -> None:
        visuals = [
            self._axis_lines,
            self._tick_lines,
            *self._tick_labels.values(),
            *self._axis_labels.values(),
        ]
        for visual in visuals:
            if visual is not None:
                visual.visible = visible

    def _update_text_visual(self, visual, *, texts, positions) -> None:
        if not texts:
            visual.text = []
            visual.visible = False
            return
        visual.text = texts
        visual.pos = np.asarray(positions, dtype=np.float32)
        visual.visible = True

    def _format_tick_value(self, value: float, lo: float, hi: float, tick_count: int) -> str:
        span = abs(float(hi) - float(lo))
        if span < 1e-12:
            return f"{value:.3g}"
        step = span / max(1.0, float(max(1, tick_count - 1)))
        decimals = 0
        rounded = round(step)
        if abs(step - rounded) > 1e-9:
            import math

            decimals = min(6, max(1, int(math.ceil(-math.log10(abs(step - rounded)))) + 1))
        elif step < 1.0:
            import math

            decimals = min(6, max(0, int(math.ceil(-math.log10(step)))))
        text = f"{value:.{decimals}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text

    def _with_alpha(self, color, alpha: float):
        rgba = list(Color(color).rgba)
        rgba[3] = min(1.0, max(0.0, float(alpha)))
        return tuple(rgba)

    def set_axes_geometry(
        self,
        *,
        render_axes,
        axes_in_middle,
        tick_count,
        tick_length_scale,
        axis_labels,
        x,
        y,
        z,
    ) -> None:
        if not render_axes:
            self._set_visible(False)
            return

        self._set_visible(True)
        tick_count = max(0, int(tick_count))

        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(np.min(y)), float(np.max(y))
        zmin, zmax = float(np.min(z)), float(np.max(z))
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)
        zmid = 0.5 * (zmin + zmax)

        axis_y = ymid if axes_in_middle else ymin
        axis_z = zmid if axes_in_middle else zmin
        axis_x = xmid if axes_in_middle else xmin

        axis_segments = np.asarray(
            [
                [xmin, axis_y, axis_z],
                [xmax, axis_y, axis_z],
                [axis_x, ymin, axis_z],
                [axis_x, ymax, axis_z],
                [axis_x, axis_y, zmin],
                [axis_x, axis_y, zmax],
            ],
            dtype=np.float32,
        )
        axis_visual = self._ensure_line_visual(attr_name="_axis_lines", width=2.0)
        axis_visual.set_data(pos=axis_segments, connect="segments")
        axis_visual.visible = True

        xtick = 0.03 * max(ymax - ymin, 1e-6) * tick_length_scale
        ytick = 0.03 * max(xmax - xmin, 1e-6) * tick_length_scale
        ztick = 0.03 * max(xmax - xmin, 1e-6) * tick_length_scale
        xoff = 0.09 * max(ymax - ymin, 1e-6)
        yoff = 0.07 * max(xmax - xmin, 1e-6)
        zoff = 0.07 * max(xmax - xmin, 1e-6)

        tick_segments: list[list[float]] = []
        x_tick_labels: list[str] = []
        y_tick_labels: list[str] = []
        z_tick_labels: list[str] = []
        x_tick_positions: list[list[float]] = []
        y_tick_positions: list[list[float]] = []
        z_tick_positions: list[list[float]] = []

        if tick_count > 0:
            for xv in np.linspace(xmin, xmax, tick_count):
                tick_segments.extend(
                    [
                        [xv, axis_y - xtick, axis_z],
                        [xv, axis_y + xtick, axis_z],
                    ]
                )
                x_tick_labels.append(self._format_tick_value(float(xv), xmin, xmax, tick_count))
                x_tick_positions.append([xv, axis_y - xoff, axis_z])
            for yv in np.linspace(ymin, ymax, tick_count):
                tick_segments.extend(
                    [
                        [axis_x - ytick, yv, axis_z],
                        [axis_x + ytick, yv, axis_z],
                    ]
                )
                y_tick_labels.append(self._format_tick_value(float(yv), ymin, ymax, tick_count))
                y_tick_positions.append([axis_x - yoff, yv, axis_z])
            for zv in np.linspace(zmin, zmax, tick_count):
                tick_segments.extend(
                    [
                        [axis_x - ztick, axis_y, zv],
                        [axis_x + ztick, axis_y, zv],
                    ]
                )
                z_tick_labels.append(self._format_tick_value(float(zv), zmin, zmax, tick_count))
                z_tick_positions.append([axis_x + zoff, axis_y, zv])

        tick_visual = self._ensure_line_visual(attr_name="_tick_lines", width=1.0)
        if tick_segments:
            tick_visual.set_data(pos=np.asarray(tick_segments, dtype=np.float32), connect="segments")
            tick_visual.visible = True
        else:
            tick_visual.visible = False

        self._update_text_visual(
            self._ensure_tick_label_visual("x"),
            texts=x_tick_labels,
            positions=x_tick_positions,
        )
        self._update_text_visual(
            self._ensure_tick_label_visual("y"),
            texts=y_tick_labels,
            positions=y_tick_positions,
        )
        self._update_text_visual(
            self._ensure_tick_label_visual("z"),
            texts=z_tick_labels,
            positions=z_tick_positions,
        )

        label_specs = {
            "x": (str(axis_labels[0]), [[xmax, axis_y - xoff * 1.8, axis_z]]),
            "y": (str(axis_labels[1]), [[axis_x - yoff * 1.8, ymax, axis_z]]),
            "z": (str(axis_labels[2]), [[axis_x + zoff * 1.8, axis_y, zmax]]),
        }
        for axis_name, (label_text, label_pos) in label_specs.items():
            label_visual = self._ensure_axis_label_visual(axis_name)
            if not label_text:
                label_visual.text = ""
                label_visual.visible = False
                continue
            label_visual.text = label_text
            label_visual.pos = np.asarray(label_pos, dtype=np.float32)
            label_visual.visible = True

    def set_axes_style(
        self,
        *,
        render_axes,
        tick_label_size,
        axis_label_size,
        axis_color,
        text_color,
        axis_alpha,
    ) -> None:
        if not render_axes:
            self._set_visible(False)
            return

        axis_rgba = self._with_alpha(axis_color, axis_alpha)
        text_rgba = self._with_alpha(text_color, axis_alpha)

        if self._axis_lines is not None:
            self._axis_lines.set_data(color=axis_rgba, width=2.0)
        if self._tick_lines is not None:
            self._tick_lines.set_data(color=axis_rgba, width=1.0)
        for visual in self._tick_labels.values():
            visual.color = text_rgba
            visual.font_size = tick_label_size
        for visual in self._axis_labels.values():
            visual.color = text_rgba
            visual.font_size = axis_label_size


class SurfaceSliceOverlay:
    def __init__(self, view):
        self.view = view
        self._line = None
        self._fill = None

    def clear(self):
        if self._line is not None:
            self._line.parent = None
            self._line = None
        if self._fill is not None:
            self._fill.parent = None
            self._fill = None

    def set_slice(self, *, axis, value, color, alpha, fill_alpha, width, x, y, z):
        rgba = list(Color(color).rgba)
        rgba[3] = min(1.0, max(0.0, float(alpha)))
        fill_rgba = list(Color(color).rgba)
        fill_rgba[3] = min(1.0, max(0.0, float(fill_alpha)))
        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(np.min(y)), float(np.max(y))
        zmin, zmax = float(np.min(z)), float(np.max(z))

        if axis == "y":
            value = min(max(float(value), ymin), ymax)
            corners = np.array(
                [
                    [xmin, value, zmin],
                    [xmax, value, zmin],
                    [xmax, value, zmax],
                    [xmin, value, zmax],
                    [xmin, value, zmin],
                ],
                dtype=np.float32,
            )
            fill_vertices = np.array(
                [
                    [xmin, value, zmin],
                    [xmax, value, zmin],
                    [xmax, value, zmax],
                    [xmin, value, zmax],
                ],
                dtype=np.float32,
            )
        else:
            value = min(max(float(value), xmin), xmax)
            corners = np.array(
                [
                    [value, ymin, zmin],
                    [value, ymax, zmin],
                    [value, ymax, zmax],
                    [value, ymin, zmax],
                    [value, ymin, zmin],
                ],
                dtype=np.float32,
            )
            fill_vertices = np.array(
                [
                    [value, ymin, zmin],
                    [value, ymax, zmin],
                    [value, ymax, zmax],
                    [value, ymin, zmax],
                ],
                dtype=np.float32,
            )

        fill_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

        if fill_rgba[3] > 0.0:
            if self._fill is None:
                self._fill = scene.visuals.Mesh(
                    vertices=fill_vertices,
                    faces=fill_faces,
                    color=tuple(fill_rgba),
                    shading=None,
                    parent=self.view.scene,
                )
                self._fill.set_gl_state(depth_test=False, blend=True, cull_face=False)
                self._fill.order = 999
            else:
                self._fill.set_data(vertices=fill_vertices, faces=fill_faces, color=tuple(fill_rgba))
        elif self._fill is not None:
            self._fill.parent = None
            self._fill = None

        if self._line is None:
            self._line = scene.visuals.Line(
                pos=corners,
                color=tuple(rgba),
                width=float(width),
                method="gl",
                parent=self.view.scene,
            )
            self._line.set_gl_state(depth_test=False, blend=True)
            self._line.order = 1000
        else:
            self._line.set_data(pos=corners, color=tuple(rgba), width=float(width))


class MorphologyRenderer:
    def __init__(self, view):
        self.view = view
        self.geometry: MorphologyGeometry | None = None
        self.collection = None
        self._color_buf = None
        self.id_colors = None
        self.id_colors_caps = None

    def clear(self) -> None:
        if self.collection is not None:
            self.collection.parent = None
            self.collection = None
        self.geometry = None
        self._color_buf = None
        self.id_colors = None
        self.id_colors_caps = None

    def set_geometry(self, geometry: MorphologyGeometry) -> None:
        self.geometry = geometry
        if self.collection is not None:
            self.collection.parent = None
        colors = geometry.colors
        if colors is None:
            colors = np.tile(np.array([0.7, 0.7, 0.7, 1.0], dtype=np.float32), (len(geometry.entity_ids), 1))

        t0 = time.perf_counter()
        self.collection = CappedCylinderCollection(
            positions=geometry.positions,
            radii=geometry.radii,
            heights=geometry.lengths,
            orientations=geometry.orientations,
            colors=colors,
            cylinder_segments=32,
            disk_slices=32,
            parent=self.view.scene,
        )
        self.collection._side_mesh.shading = None
        self.collection._cap_mesh.shading = None

        n = len(geometry.entity_ids)
        self._color_buf = np.empty((n, 4), dtype=np.float32)
        self._color_buf[:, 1] = 0.2
        self._color_buf[:, 3] = 1.0

        def make_id_color(i):
            cid = i + 1
            return np.array(
                [
                    (cid & 0xFF) / 255.0,
                    ((cid >> 8) & 0xFF) / 255.0,
                    ((cid >> 16) & 0xFF) / 255.0,
                    1.0,
                ],
                dtype=np.float32,
            )

        self.id_colors = np.stack([make_id_color(i) for i in range(n)], axis=0)
        self.id_colors_caps = np.vstack([self.id_colors, self.id_colors])
        elapsed = time.perf_counter() - t0
        print(f"Morphology visual generated in {elapsed:.2f}s")

    def pick(self, xf, yf, canvas) -> str | None:
        if self.collection is None or self.geometry is None:
            return None
        side, cap = self.collection._side_mesh, self.collection._cap_mesh
        old_side, old_cap = side.instance_colors, cap.instance_colors
        side.instance_colors = self.id_colors
        cap.instance_colors = self.id_colors_caps
        img = canvas.render(region=(xf, yf, 1, 1), size=(1, 1), alpha=False)
        side.instance_colors, cap.instance_colors = old_side, old_cap
        idx = self._decode_pick_index(img)
        if idx is None or idx >= len(self.geometry.entity_ids):
            return None
        return self.geometry.entity_ids[idx]

    def _decode_pick_index(self, img: np.ndarray) -> int | None:
        pixels = np.asarray(img)
        if pixels.ndim != 3 or pixels.shape[2] < 3 or pixels.shape[0] == 0 or pixels.shape[1] == 0:
            return None
        if pixels.dtype != np.uint8:
            pixels = np.round(pixels * 255).astype(np.uint8)
        pix = pixels[0, 0]
        cid = int(pix[0]) | (int(pix[1]) << 8) | (int(pix[2]) << 16)
        return cid - 1 if cid > 0 else None

    def update_colors(self, data: np.ndarray, color_map: str, *, color_limits=None, color_norm: str = "auto") -> None:
        if self.collection is None:
            return
        values = np.asarray(data, dtype=np.float32)
        if color_limits is not None:
            vmin, vmax = float(color_limits[0]), float(color_limits[1])
            if abs(vmax - vmin) < 1e-12:
                norm = np.zeros_like(values, dtype=np.float32)
            else:
                norm = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
        elif color_norm == "symmetric":
            vmax = float(np.max(np.abs(values)))
            if vmax < 1e-12:
                norm = np.full_like(values, 0.5, dtype=np.float32)
            else:
                norm = np.clip((values + vmax) / (2.0 * vmax), 0.0, 1.0)
        else:
            vmin = float(np.min(values))
            vmax = float(np.max(values))
            if abs(vmax - vmin) < 1e-12:
                norm = np.zeros_like(values, dtype=np.float32)
            else:
                norm = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
        if str(color_map).strip().lower() == "scalar":
            self._color_buf[:, 0] = norm
            self._color_buf[:, 1] = 0.2
            self._color_buf[:, 2] = 1.0 - norm
            self._color_buf[:, 3] = 1.0
        else:
            lut = _colormap_samples(color_map)
            idx = np.clip((norm * (len(lut) - 1)).astype(np.int32), 0, len(lut) - 1)
            self._color_buf[:, :] = lut[idx]
        self.collection.set_colors(self._color_buf)


class SurfaceRenderer:
    def __init__(self, view):
        self.view = view
        self.surface = None
        self.axes = SurfaceAxesOverlay(view)
        self._slice_overlays: dict[str, SurfaceSliceOverlay] = {}
        self._grid_shape = None
        self._last_z = None

    def clear(self) -> None:
        if self.surface is not None:
            self.surface.parent = None
            self.surface = None
        self.axes.clear()
        self.clear_operator_overlays()
        self._grid_shape = None
        self._last_z = None

    def clear_operator_overlays(self) -> None:
        for overlay in self._slice_overlays.values():
            overlay.clear()
        self._slice_overlays.clear()

    def set_slice_operator_overlays(self, overlays, *, x, y, z) -> None:
        overlay_ids = {overlay["operator_id"] for overlay in overlays}
        stale_ids = set(self._slice_overlays) - overlay_ids
        for operator_id in stale_ids:
            self._slice_overlays[operator_id].clear()
            del self._slice_overlays[operator_id]
        for overlay in overlays:
            slice_overlay = self._slice_overlays.get(overlay["operator_id"])
            if slice_overlay is None:
                slice_overlay = SurfaceSliceOverlay(self.view)
                self._slice_overlays[overlay["operator_id"]] = slice_overlay
            slice_overlay.set_slice(
                axis=overlay["axis"],
                value=overlay["value"],
                color=overlay["color"],
                alpha=overlay["alpha"],
                fill_alpha=overlay["fill_alpha"],
                width=overlay["width"],
                x=x,
                y=y,
                z=z,
            )

    def _map_height_to_colors(self, z: np.ndarray, color_map: str, color_limits) -> np.ndarray:
        if color_limits is None:
            zmin = float(np.min(z))
            zmax = float(np.max(z))
        else:
            zmin, zmax = float(color_limits[0]), float(color_limits[1])
        if abs(zmax - zmin) < 1e-12:
            norm = np.zeros_like(z, dtype=np.float32)
        else:
            norm = np.clip((z - zmin) / (zmax - zmin), 0.0, 1.0).astype(np.float32)
        lut = _colormap_samples(color_map)
        idx = np.clip((norm * (len(lut) - 1)).astype(np.int32), 0, len(lut) - 1)
        return lut[idx]

    def _surface_rgba(self, color, surface_alpha: float) -> tuple[float, float, float, float]:
        rgba = list(Color(color).rgba)
        rgba[3] = float(surface_alpha)
        return tuple(rgba)

    def _resolve_shading(self, shading):
        if shading is None:
            return None
        name = str(shading).strip().lower()
        if name in {"", "none", "unlit"}:
            return None
        if name in {"lit", "smooth"}:
            return "smooth"
        if name == "flat":
            return "flat"
        return name

    def _clear_surface_vertex_colors(self) -> None:
        if self.surface is None:
            return
        mesh_data = self.surface.mesh_data
        mesh_data._vertex_colors = None
        mesh_data._vertex_colors_indexed_by_faces = None
        mesh_data._vertex_colors_indexed_by_edges = None

    def update_surface(
        self,
        x,
        y,
        z,
        *,
        color_map,
        color_limits,
        colors,
        color_by,
        surface_color,
        surface_shading,
        surface_alpha,
        coords_changed=True,
    ):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        z = np.asarray(z, dtype=np.float32)
        self._last_z = z
        color_by_name = str(color_by).strip().lower()
        if colors is None and color_by_name == "height":
            colors = self._map_height_to_colors(z, color_map, color_limits)
        if colors is not None:
            colors = np.asarray(colors, dtype=np.float32).copy()
            if colors.shape[-1] == 4:
                colors[..., 3] = float(surface_alpha)
        surface_rgba = self._surface_rgba(surface_color, surface_alpha)
        resolved_shading = self._resolve_shading(surface_shading)
        grid_shape = (tuple(x.shape), tuple(y.shape), tuple(z.shape))
        recreate_surface = self.surface is None or self._grid_shape != grid_shape
        if recreate_surface:
            if self.surface is not None:
                self.surface.parent = None
            self.surface = scene.visuals.SurfacePlot(
                x=x,
                y=y,
                z=z,
                color=surface_rgba,
                shading=resolved_shading,
                parent=self.view.scene,
            )
            self.surface.set_gl_state("translucent", depth_test=True, cull_face=False)
            self._grid_shape = grid_shape
        else:
            self.surface.shading = resolved_shading
        if coords_changed:
            # Grid shape is the same but x/y coordinates changed — full data upload.
            if colors is not None:
                self.surface.set_data(x=x, y=y, z=z, colors=colors)
            else:
                self.surface.set_data(x=x, y=y, z=z)
                self.surface.color = surface_rgba
        else:
            # Only z (and colors) changed — skip x/y GPU upload.
            if colors is not None:
                self.surface.set_data(z=z, colors=colors)
            else:
                self.surface.set_data(z=z)
                self.surface.color = surface_rgba
        if recreate_surface:
            self.view.camera.set_range()

    def update_surface_style(
        self,
        z,
        *,
        color_map,
        color_limits,
        colors,
        color_by,
        surface_color,
        surface_shading,
        surface_alpha,
    ) -> None:
        if self.surface is None:
            return
        z = np.asarray(z, dtype=np.float32)
        self._last_z = z
        color_by_name = str(color_by).strip().lower()
        surface_rgba = self._surface_rgba(surface_color, surface_alpha)
        self.surface.shading = self._resolve_shading(surface_shading)
        if colors is None and color_by_name == "height":
            colors = self._map_height_to_colors(z, color_map, color_limits)
        if colors is not None:
            colors = np.asarray(colors, dtype=np.float32).copy()
            if colors.shape[-1] == 4:
                colors[..., 3] = float(surface_alpha)
            self.surface.set_data(colors=colors)
        else:
            self._clear_surface_vertex_colors()
            self.surface.color = surface_rgba
