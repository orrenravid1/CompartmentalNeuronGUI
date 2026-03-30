from __future__ import annotations

import time

import numpy as np
from vispy import scene
from vispy.color import Color

from compneurovis.core.geometry import MorphologyGeometry
from compneurovis.vispyutils.cappedcylindercollection import CappedCylinderCollection


class SurfaceAxesOverlay:
    def __init__(self, view):
        self.view = view
        self._visuals = []

    def clear(self) -> None:
        for vis in self._visuals:
            vis.parent = None
        self._visuals.clear()

    def _add_line(self, points, color, width=2):
        line = scene.visuals.Line(
            pos=np.asarray(points, dtype=np.float32),
            color=color,
            width=width,
            method="gl",
            parent=self.view.scene,
        )
        line.set_gl_state(depth_test=False, blend=True)
        line.order = 1000
        self._visuals.append(line)

    def _add_text(self, text, pos, color, font_size=10, anchor_x="center", anchor_y="center"):
        label = scene.visuals.Text(
            text=str(text),
            pos=np.asarray(pos, dtype=np.float32),
            color=color,
            font_size=font_size,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
            parent=self.view.scene,
        )
        label.set_gl_state(depth_test=False, blend=True)
        label.order = 1000
        self._visuals.append(label)

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

    def set_axes(
        self,
        *,
        render_axes,
        axes_in_middle,
        tick_count,
        tick_length_scale,
        tick_label_size,
        axis_label_size,
        axis_color,
        text_color,
        axis_alpha,
        axis_labels,
        x,
        y,
        z,
    ):
        self.clear()
        if not render_axes:
            return

        axis_color = self._with_alpha(axis_color, axis_alpha)
        text_color = self._with_alpha(text_color, axis_alpha)
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

        self._add_line([[xmin, axis_y, axis_z], [xmax, axis_y, axis_z]], axis_color)
        self._add_line([[axis_x, ymin, axis_z], [axis_x, ymax, axis_z]], axis_color)
        self._add_line([[axis_x, axis_y, zmin], [axis_x, axis_y, zmax]], axis_color)

        xtick = 0.03 * max(ymax - ymin, 1e-6) * tick_length_scale
        ytick = 0.03 * max(xmax - xmin, 1e-6) * tick_length_scale
        ztick = 0.03 * max(xmax - xmin, 1e-6) * tick_length_scale
        xoff = 0.09 * max(ymax - ymin, 1e-6)
        yoff = 0.07 * max(xmax - xmin, 1e-6)
        zoff = 0.07 * max(xmax - xmin, 1e-6)

        if tick_count > 0:
            for xv in np.linspace(xmin, xmax, tick_count):
                self._add_line([[xv, axis_y - xtick, axis_z], [xv, axis_y + xtick, axis_z]], axis_color, width=1)
                self._add_text(
                    self._format_tick_value(float(xv), xmin, xmax, tick_count),
                    [xv, axis_y - xoff, axis_z],
                    text_color,
                    font_size=tick_label_size,
                    anchor_y="top",
                )
            for yv in np.linspace(ymin, ymax, tick_count):
                self._add_line([[axis_x - ytick, yv, axis_z], [axis_x + ytick, yv, axis_z]], axis_color, width=1)
                self._add_text(
                    self._format_tick_value(float(yv), ymin, ymax, tick_count),
                    [axis_x - yoff, yv, axis_z],
                    text_color,
                    font_size=tick_label_size,
                    anchor_x="right",
                )
            for zv in np.linspace(zmin, zmax, tick_count):
                self._add_line([[axis_x - ztick, axis_y, zv], [axis_x + ztick, axis_y, zv]], axis_color, width=1)
                self._add_text(
                    self._format_tick_value(float(zv), zmin, zmax, tick_count),
                    [axis_x + zoff, axis_y, zv],
                    text_color,
                    font_size=tick_label_size,
                    anchor_x="left",
                )

        self._add_text(axis_labels[0], [xmax, axis_y - xoff * 1.8, axis_z], text_color, font_size=axis_label_size, anchor_y="top")
        self._add_text(axis_labels[1], [axis_x - yoff * 1.8, ymax, axis_z], text_color, font_size=axis_label_size, anchor_x="right")
        self._add_text(axis_labels[2], [axis_x + zoff * 1.8, axis_y, zmax], text_color, font_size=axis_label_size, anchor_x="left")


class SurfaceSliceOverlay:
    def __init__(self, view):
        self.view = view
        self._line = None

    def clear(self):
        if self._line is not None:
            self._line.parent = None
            self._line = None

    def set_slice(self, *, axis, value, color, alpha, width, x, y, z):
        rgba = list(Color(color).rgba)
        rgba[3] = min(1.0, max(0.0, float(alpha)))
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

        if self._line is not None:
            self._line.parent = None
            self._line = None

        self._line = scene.visuals.Line(
            pos=corners,
            color=tuple(rgba),
            width=float(width),
            method="gl",
            parent=self.view.scene,
        )
        self._line.set_gl_state(depth_test=False, blend=True)
        self._line.order = 1000


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

    def update_colors(self, data: np.ndarray, color_map: str) -> None:
        if self.collection is None:
            return
        values = np.asarray(data, dtype=np.float32)
        if color_map == "voltage":
            norm = np.clip((values + 80.0) / 130.0, 0.0, 1.0)
        else:
            vmin = float(np.min(values))
            vmax = float(np.max(values))
            if abs(vmax - vmin) < 1e-12:
                norm = np.zeros_like(values, dtype=np.float32)
            else:
                norm = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
        self._color_buf[:, 0] = norm
        self._color_buf[:, 2] = 1.0 - norm
        self.collection.set_colors(self._color_buf)


class SurfaceRenderer:
    def __init__(self, view):
        self.view = view
        self.surface = None
        self.axes = SurfaceAxesOverlay(view)
        self.slice_overlay = SurfaceSliceOverlay(view)
        self._grid_shape = None

    def clear(self) -> None:
        if self.surface is not None:
            self.surface.parent = None
            self.surface = None
        self.axes.clear()
        self.slice_overlay.clear()
        self._grid_shape = None

    def _colormap_samples(self, name: str, n: int = 256) -> np.ndarray:
        name = str(name).lower()
        x = np.linspace(0.0, 1.0, n, dtype=np.float32)
        if name == "grayscale":
            rgb = np.stack([x, x, x], axis=1)
        elif name == "fire":
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

    def _map_height_to_colors(self, z: np.ndarray, cmap: str, clim) -> np.ndarray:
        if clim is None:
            zmin = float(np.min(z))
            zmax = float(np.max(z))
        else:
            zmin, zmax = float(clim[0]), float(clim[1])
        if abs(zmax - zmin) < 1e-12:
            norm = np.zeros_like(z, dtype=np.float32)
        else:
            norm = np.clip((z - zmin) / (zmax - zmin), 0.0, 1.0).astype(np.float32)
        lut = self._colormap_samples(cmap)
        idx = np.clip((norm * (len(lut) - 1)).astype(np.int32), 0, len(lut) - 1)
        return lut[idx]

    def update_surface(self, x, y, z, *, cmap, clim, colors, color_by, surface_alpha, coords_changed=True):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        z = np.asarray(z, dtype=np.float32)
        if colors is None and color_by == "height":
            colors = self._map_height_to_colors(z, cmap, clim)
        if colors is not None:
            colors = np.asarray(colors, dtype=np.float32).copy()
            if colors.shape[-1] == 4:
                colors[..., 3] = float(surface_alpha)
        grid_shape = (tuple(x.shape), tuple(y.shape), tuple(z.shape))
        recreate_surface = self.surface is None or self._grid_shape != grid_shape
        if recreate_surface:
            if self.surface is not None:
                self.surface.parent = None
            self.surface = scene.visuals.SurfacePlot(
                x=x,
                y=y,
                z=z,
                color=(0.5, 0.6, 0.8, surface_alpha),
                shading=None,
                parent=self.view.scene,
            )
            self.surface.set_gl_state("translucent", depth_test=True, cull_face=False)
            self._grid_shape = grid_shape
        elif coords_changed:
            # Grid shape is the same but x/y coordinates changed — full data upload.
            if colors is not None:
                self.surface.set_data(x=x, y=y, z=z, colors=colors)
            else:
                self.surface.set_data(x=x, y=y, z=z, color=(0.5, 0.6, 0.8, surface_alpha))
        else:
            # Only z (and colors) changed — skip x/y GPU upload.
            if colors is not None:
                self.surface.set_data(z=z, colors=colors)
            else:
                self.surface.set_data(z=z, color=(0.5, 0.6, 0.8, surface_alpha))
        if recreate_surface:
            self.view.camera.set_range()
