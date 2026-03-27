import time
import multiprocessing as mp
from multiprocessing import Pipe, Process
from collections import deque
from typing import TypedDict
import numpy as np
import sys

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import Qt
import pyqtgraph as pg

from vispy import use
use(app='pyqt6', gl='gl+')
from vispy import scene, app as vispy_app
from vispy.scene.cameras import TurntableCamera

from compneurovis.vispyutils.cappedcylindercollection import CappedCylinderCollection
from compneurovis.simulation import Simulation, simulation_process

TRACE_COLORS = ['b', 'r', 'g', 'm', 'c', 'y', (255,165,0), (128,0,128)]


class MorphologyMeta(TypedDict):
    kind: str
    positions: np.ndarray
    orientations: np.ndarray
    radii: np.ndarray
    lengths: np.ndarray
    colors: np.ndarray
    sec_names: list[str]
    sec_idx: np.ndarray
    xloc: np.ndarray


class SurfaceMeta(TypedDict, total=False):
    kind: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    colors: np.ndarray
    color_by: str
    cmap: str
    clim: tuple[float, float]
    surface_alpha: float
    background_color: str | tuple[float, float, float] | tuple[float, float, float, float]
    render_axes: bool
    axes_in_middle: bool
    tick_count: int
    axis_color: str | tuple[float, float, float] | tuple[float, float, float, float]
    axis_labels: tuple[str, str, str]
    title: str


class SurfaceAxesOverlay:
    def __init__(self, view):
        self.view = view
        self._visuals = []

    def clear(self):
        for vis in self._visuals:
            vis.parent = None
        self._visuals.clear()

    def _add_line(self, points, color, width=2):
        line = scene.visuals.Line(
            pos=np.asarray(points, dtype=np.float32),
            color=color,
            width=width,
            method='gl',
            parent=self.view.scene,
        )
        self._visuals.append(line)

    def _add_text(self, text, pos, color, font_size=10, anchor_x='center', anchor_y='center'):
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

    def set_axes(self, meta, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        self.clear()
        if not meta.get('render_axes', False):
            return

        axis_color = meta.get('axis_color', 'black')
        tick_count = max(0, int(meta.get('tick_count', 5)))
        centered = bool(meta.get('axes_in_middle', True))
        axis_labels = meta.get('axis_labels', ('x', 'y', 'z'))

        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(np.min(y)), float(np.max(y))
        zmin, zmax = float(np.min(z)), float(np.max(z))
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)
        zmid = 0.5 * (zmin + zmax)

        axis_y = ymid if centered else ymin
        axis_z = zmid if centered else zmin
        axis_x = xmid if centered else xmin

        self._add_line([[xmin, axis_y, axis_z], [xmax, axis_y, axis_z]], axis_color)
        self._add_line([[axis_x, ymin, axis_z], [axis_x, ymax, axis_z]], axis_color)
        self._add_line([[axis_x, axis_y, zmin], [axis_x, axis_y, zmax]], axis_color)

        xtick = 0.03 * max(ymax - ymin, 1e-6)
        ytick = 0.03 * max(xmax - xmin, 1e-6)
        ztick = 0.03 * max(xmax - xmin, 1e-6)
        xoff = 0.09 * max(ymax - ymin, 1e-6)
        yoff = 0.07 * max(xmax - xmin, 1e-6)
        zoff = 0.07 * max(xmax - xmin, 1e-6)

        if tick_count > 0:
            for xv in np.linspace(xmin, xmax, tick_count):
                self._add_line([[xv, axis_y - xtick, axis_z], [xv, axis_y + xtick, axis_z]], axis_color, width=1)
                self._add_text(f"{xv:.0f}", [xv, axis_y - xoff, axis_z], axis_color, font_size=12, anchor_y='top')
            for yv in np.linspace(ymin, ymax, tick_count):
                self._add_line([[axis_x - ytick, yv, axis_z], [axis_x + ytick, yv, axis_z]], axis_color, width=1)
                self._add_text(f"{yv:.0f}", [axis_x - yoff, yv, axis_z], axis_color, font_size=12, anchor_x='right')
            for zv in np.linspace(zmin, zmax, tick_count):
                self._add_line([[axis_x - ztick, axis_y, zv], [axis_x + ztick, axis_y, zv]], axis_color, width=1)
                self._add_text(f"{zv:.0f}", [axis_x + zoff, axis_y, zv], axis_color, font_size=12, anchor_x='left')

        self._add_text(axis_labels[0], [xmax, axis_y - xoff * 1.8, axis_z], axis_color, font_size=16, anchor_y='top')
        self._add_text(axis_labels[1], [axis_x - yoff * 1.8, ymax, axis_z], axis_color, font_size=16, anchor_x='right')
        self._add_text(axis_labels[2], [axis_x + zoff * 1.8, axis_y, zmax], axis_color, font_size=16, anchor_x='left')

class _Trace:
    """A single recorded trace on the plot."""
    __slots__ = ('sec', 'xloc', 'seg_idx', 'varname', 'plot_item', 't', 'v')

    def __init__(self, sec, xloc, seg_idx, plot_item, varname='v'):
        self.sec = sec
        self.xloc = xloc
        self.seg_idx = seg_idx       # int for morph vars, None for scalar sim vars
        self.varname = varname       # key into data dict
        self.plot_item = plot_item
        self.t = deque(maxlen=1000)
        self.v = deque(maxlen=1000)


class MorphologyManager:
    """VisPy instancing, picking & color‐mapping from flat arrays."""
    def __init__(self, view):
        self.view = view

    def set_morphology(self, meta):
        pos   = meta['positions']
        ori   = meta['orientations']
        rad   = meta['radii']
        ln    = meta['lengths']
        col   = meta['colors']
        names = meta['sec_names']
        idx   = meta['sec_idx']
        xlocs = meta['xloc']

        self.sec_names = names
        self.sec_idx   = idx
        self.xlocs     = xlocs

        t0 = time.perf_counter()
        self.collection = CappedCylinderCollection(
            positions=pos, radii=rad, heights=ln,
            orientations=ori, colors=col,
            cylinder_segments=32, disk_slices=32,
            parent=self.view.scene
        )
        self.collection._side_mesh.shading = None
        self.collection._cap_mesh.shading  = None

        N = pos.shape[0]
        self._color_buf = np.empty((N, 4), dtype=np.float32)
        self._color_buf[:,1] = 0.2
        self._color_buf[:,3] = 1.0

        def make_id_color(i):
            cid = i+1
            return np.array([
                (cid        & 0xFF)/255.0,
                ((cid >>  8) & 0xFF)/255.0,
                ((cid >> 16) & 0xFF)/255.0,
                1.0
            ], dtype=np.float32)

        self.id_colors      = np.stack([make_id_color(i) for i in range(N)], axis=0)
        self.id_colors_caps = np.vstack([self.id_colors, self.id_colors])

        elapsed = time.perf_counter() - t0
        print(f"Morphology visual generated in {elapsed:.2f}s")

    def pick(self, xf, yf, canvas):
        side, cap = self.collection._side_mesh, self.collection._cap_mesh
        old_side, old_cap = side.instance_colors, cap.instance_colors

        side.instance_colors = self.id_colors
        cap.instance_colors  = self.id_colors_caps
        img = canvas.render(region=(xf, yf, 1,1), size=(1,1), alpha=False)
        side.instance_colors, cap.instance_colors = old_side, old_cap

        pix = img[0,0]
        if pix.dtype != np.uint8:
            pix = np.round(pix*255).astype(int)
        cid = int(pix[0]) | (int(pix[1])<<8) | (int(pix[2])<<16)
        idx = cid - 1 if cid>0 else None
        if idx is None or idx >= len(self.sec_idx):
            return None
        sec  = self.sec_names[self.sec_idx[idx]]
        xloc = self.xlocs[idx]
        return sec, xloc

    def update_colors(self, data, map_fn):
        norm = map_fn(data)
        self._color_buf[:,0] = norm
        self._color_buf[:,2] = 1.0 - norm
        self.collection.set_colors(self._color_buf)


class SurfaceManager:
    def __init__(self, view):
        self.view = view
        self.surface = None
        self.axes = SurfaceAxesOverlay(view)

    def _colormap_samples(self, name: str, n: int = 256) -> np.ndarray:
        name = str(name).lower()
        x = np.linspace(0.0, 1.0, n, dtype=np.float32)
        if name == 'grayscale':
            rgb = np.stack([x, x, x], axis=1)
        elif name == 'fire':
            rgb = np.stack([
                np.clip(1.5 * x, 0.0, 1.0),
                np.clip(2.0 * x - 0.4, 0.0, 1.0),
                np.clip(4.0 * x - 3.0, 0.0, 1.0),
            ], axis=1)
        else:
            # blue-white-red default
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

    def _map_height_to_colors(self, z: np.ndarray, meta) -> np.ndarray:
        clim = meta.get('clim')
        if clim is None:
            zmin = float(np.min(z))
            zmax = float(np.max(z))
        else:
            zmin = float(clim[0])
            zmax = float(clim[1])
        if abs(zmax - zmin) < 1e-12:
            norm = np.zeros_like(z, dtype=np.float32)
        else:
            norm = np.clip((z - zmin) / (zmax - zmin), 0.0, 1.0).astype(np.float32)
        lut = self._colormap_samples(meta.get('cmap', 'bwr'))
        idx = np.clip((norm * (len(lut) - 1)).astype(np.int32), 0, len(lut) - 1)
        return lut[idx]

    def set_surface(self, meta):
        x = np.asarray(meta['x'], dtype=np.float32)
        y = np.asarray(meta['y'], dtype=np.float32)
        z = np.asarray(meta['z'], dtype=np.float32)
        colors = meta.get('colors')
        color_by = meta.get('color_by')
        surface_alpha = float(meta.get('surface_alpha', 1.0))
        surface_alpha = min(1.0, max(0.0, surface_alpha))

        if self.surface is not None:
            self.surface.parent = None
            self.surface = None

        if colors is None and color_by == 'height':
            colors = self._map_height_to_colors(z, meta)
        if colors is not None:
            colors = np.asarray(colors, dtype=np.float32).copy()
            if colors.shape[-1] == 4:
                colors[..., 3] = surface_alpha

        self.surface = scene.visuals.SurfacePlot(
            x=x,
            y=y,
            z=z,
            color=(0.5, 0.6, 0.8, surface_alpha),
            shading=None,
            parent=self.view.scene,
        )
        self.surface.set_gl_state('translucent', depth_test=True, cull_face=False)

        if colors is not None:
            self.surface.set_data(z=z, colors=np.asarray(colors, dtype=np.float32))

        self.axes.set_axes(meta, x, y, z)
        self.view.camera.set_range()


def _payload_kind(payload) -> str | None:
    if not isinstance(payload, dict):
        return None
    kind = payload.get('kind')
    if isinstance(kind, str):
        return kind
    morphology_keys = {
        'positions', 'orientations', 'radii', 'lengths',
        'colors', 'sec_names', 'sec_idx', 'xloc'
    }
    if morphology_keys.issubset(payload.keys()):
        return 'morphology'
    surface_keys = {'x', 'y', 'z'}
    if surface_keys.issubset(payload.keys()):
        return 'surface'
    return None


class SimulationViewer(QtWidgets.QMainWindow):
    def __init__(self, sim: Simulation):
        super().__init__()
        self.setWindowTitle("Simulation Viewer")
        self.statusBar().showMessage("Loading simulation…")

        # 3D canvas
        self.canvas3d = scene.SceneCanvas(keys='interactive',
                                          bgcolor='white', show=False)
        self.view     = self.canvas3d.central_widget.add_view()
        self.view.camera = TurntableCamera(fov=60, distance=200,
                                           elevation=30, azimuth=30,
                                           translate_speed=100, up='+z')

        self.sim            = sim
        self.mgr            = MorphologyManager(self.view)
        self.surface_mgr    = SurfaceManager(self.view)
        self.morphology_ready = False
        self._color_var_key = None

        # 2D plot
        self.plot2d = pg.PlotWidget(title="Traces")
        self.plot2d.setLabel('bottom','Time','ms')
        self.plot2d.setLabel('left','Value','')
        self.plot2d.setBackground('w')
        self.plot2d.addLegend(offset=(10, 10))
        self.traces: list[_Trace] = []
        vb = self.plot2d.getPlotItem().getViewBox()
        # fixed y-range for voltage, allow x auto-range
        self._vb_ymin, self._vb_ymax = -120, 120
        vb.setLimits(yMin=self._vb_ymin, yMax=self._vb_ymax)
        vb.setRange(yRange=(self._vb_ymin, self._vb_ymax), padding=0)
        vb.enableAutoRange(x=True, y=False)

        # layout
        w = QtWidgets.QWidget()
        hb = QtWidgets.QHBoxLayout(w)
        hb.addWidget(self.canvas3d.native)

        # Right side: plot + controls
        right_w = QtWidgets.QWidget()
        right_l = QtWidgets.QVBoxLayout(right_w)
        right_l.setContentsMargins(0,0,0,0)
        right_l.addWidget(self.plot2d)

        # Parameter controls: dynamically generated from simulation.spec
        ctrl_w = QtWidgets.QWidget()
        ctrl_l = QtWidgets.QVBoxLayout(ctrl_w)
        ctrl_l.setContentsMargins(6,6,6,6)
        ctrl_l.setSpacing(6)
        self._controls = {}
        try:
            specs = sim.controllable_parameters() or {}
        except Exception as e:
            specs = {}
            print(f"Error retrieving controllable parameters: {e}")
        
        print (specs)

        for name, spec in specs.items():
            h = QtWidgets.QWidget()
            hl = QtWidgets.QHBoxLayout(h)
            hl.setContentsMargins(0,0,0,0)
            lab = QtWidgets.QLabel(spec.get('label', name))
            hl.addWidget(lab)
            t = spec.get('type', 'float')
            if t in ('float', 'double'):
                steps = int(spec.get('steps', 100))
                mn = float(spec.get('min', 0.0))
                mx = float(spec.get('max', 1.0))
                scale = spec.get('scale', 'linear')
                slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
                slider.setRange(0, steps)
                # initial value
                init = spec.get('default', 0.5 * (mx - mn) + mn)
                print(f"Control {name}: default={init}")
                try:
                    if scale == 'log' and mn > 0 and mx > mn:
                        import math
                        lo = math.log(mn)
                        hi = math.log(mx)
                        v0 = int(round((math.log(float(init)) - lo) / (hi - lo) * steps))
                    else:
                        v0 = int(round((float(init) - mn) / (mx - mn) * steps))
                except Exception:
                    v0 = 0
                slider.setValue(max(0, min(steps, v0)))
                value_label = QtWidgets.QLabel("")

                def make_float_cb(n, mn, mx, steps, lbl, scale):
                    def _cb(val: int):
                        frac = (val / steps)
                        try:
                            if scale == 'log' and mn > 0 and mx > mn:
                                # geometric interpolation
                                v = mn * ((mx / mn) ** frac)
                            else:
                                v = mn + (mx - mn) * frac
                            lbl.setText(f"{v:.3g}")
                            self.sim.on_control_gui(n, float(v))
                            try:
                                self.cmd_parent.send(("control", n, float(v)))
                            except Exception:
                                pass
                        except Exception:
                            pass
                    return _cb

                slider.valueChanged.connect(make_float_cb(name, mn, mx, steps, value_label, scale))
                # trigger display update for initial
                slider.valueChanged.emit(slider.value())
                hl.addWidget(slider, 1)
                hl.addWidget(value_label)
                self._controls[name] = slider

            elif t == 'int':
                mn = int(spec.get('min', 0))
                mx = int(spec.get('max', 100))
                spin = QtWidgets.QSpinBox()
                spin.setRange(mn, mx)
                spin.setValue(int(spec.get('default', getattr(sim, name, mn))))
                def make_int_cb(n):
                    def _cb(val: int):
                        try:
                            self.cmd_parent.send(("control", n, int(val)))
                        except Exception:
                            pass
                    return _cb
                spin.valueChanged.connect(make_int_cb(name))
                hl.addWidget(spin)
                self._controls[name] = spin

            elif t == 'bool':
                cb = QtWidgets.QCheckBox()
                cb.setChecked(bool(spec.get('default', getattr(sim, name, False))))
                def make_bool_cb(n):
                    def _cb(val: bool):
                        try:
                            self.cmd_parent.send(("control", n, bool(val)))
                        except Exception:
                            pass
                    return _cb
                cb.toggled.connect(make_bool_cb(name))
                hl.addWidget(cb)
                self._controls[name] = cb

            elif t == 'enum':
                opts = list(spec.get('options', []))
                combo = QtWidgets.QComboBox()
                combo.addItems([str(o) for o in opts])
                default = spec.get('default', getattr(sim, name, None))
                if default in opts:
                    combo.setCurrentIndex(opts.index(default))
                def make_enum_cb(n, options):
                    def _cb(idx: int):
                        try:
                            self.cmd_parent.send(("control", n, options[int(idx)]))
                        except Exception:
                            pass
                    return _cb
                combo.currentIndexChanged.connect(make_enum_cb(name, opts))
                hl.addWidget(combo)
                self._controls[name] = combo

            else:
                # Unknown type: show a label of the default
                val_lab = QtWidgets.QLabel(str(spec.get('default', getattr(sim, name, ''))))
                hl.addWidget(val_lab)
                self._controls[name] = val_lab

            ctrl_l.addWidget(h)

        right_l.addWidget(ctrl_w)

        hb.addWidget(right_w)
        self.setCentralWidget(w)
        self.resize(1200,600)

        # pipes
        self.data_parent, self.data_child = Pipe(duplex=True)
        self.cmd_parent,  self.cmd_child  = Pipe(duplex=True)

        # start worker
        self.worker = Process(
            target=simulation_process,
            args=(sim, self.data_child, self.cmd_child)
        )
        QtCore.QTimer.singleShot(0, self._start_worker)

        # polling
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(1000 // 60)

        self.selected = None
        self.DRAG_THRESHOLD = 5
        self.has_morphology = False
        self.scene_kind = None

    def _start_worker(self):
        self._load_t0 = time.perf_counter()
        self.worker.start()
        self.data_child.close()
        self.cmd_child.close()

    def _handle_initial_payload(self, payload):
        elapsed = time.perf_counter() - self._load_t0
        kind = _payload_kind(payload)
        self.scene_kind = kind
        bg = payload.get("background_color") if isinstance(payload, dict) else None
        if bg is not None:
            self.canvas3d.bgcolor = bg
        if kind == "morphology":
            self.canvas3d.native.show()
            self.mgr.set_morphology(payload)
            self.has_morphology = True
            self.morphology_ready = True
            self._color_var_key = None
            self.statusBar().showMessage(f"Loaded morphology in {elapsed:.2f}s")
            if not self.traces:
                self.select(self.mgr.sec_names[0], 0.5)
            self.canvas3d.events.mouse_press.connect(self._on_mouse_press)
            self.canvas3d.events.mouse_release.connect(self._on_mouse_release)
        elif kind == "surface":
            self.canvas3d.native.show()
            self.surface_mgr.set_surface(payload)
            self.has_morphology = False
            self.morphology_ready = False
            title = payload.get("title", "surface")
            self.statusBar().showMessage(f"Loaded {title} in {elapsed:.2f}s")
        else:
            self.has_morphology = False
            self.morphology_ready = False
            self.canvas3d.native.hide()
            self.statusBar().showMessage(f"Loaded simulation in {elapsed:.2f}s")

        print(f"Loaded in {elapsed:.2f}s")
        QtCore.QTimer.singleShot(3000, self.statusBar().clearMessage)

    def _poll(self):
        try:
            colors_dirty = False
            while self.data_parent.poll():
                msg = self.data_parent.recv()
                if isinstance(msg, tuple) and len(msg) == 2:
                    kind, payload = msg
                else:
                    kind, payload = "data", msg

                if kind == "initial_payload":
                    self._handle_initial_payload(payload)
                    continue
                if kind == "scene_payload":
                    self._handle_initial_payload(payload)
                    self.canvas3d.update()
                    continue

                # subsequent messages: data dict (or legacy (t, v) tuple)
                if isinstance(payload, dict):
                    data = payload
                    t = data['t']
                else:
                    t, arr = payload
                    data = {'t': t, 'v': arr}

                # 3D color mapping — find color key once, reuse thereafter
                if self.has_morphology and self._color_var_key is None:
                    if 'v' in data and isinstance(data['v'], np.ndarray):
                        self._color_var_key = 'v'
                    else:
                        for k, val in data.items():
                            if k != 't' and isinstance(val, np.ndarray) and val.ndim == 1:
                                self._color_var_key = k
                                break

                color_arr = data.get(self._color_var_key) if self.has_morphology and self._color_var_key else None
                if color_arr is not None:
                    if self._color_var_key == 'v':
                        self.mgr.update_colors(color_arr, lambda a: np.clip((a+80)/130,0,1))
                    else:
                        self.mgr.update_colors(color_arr, lambda a: np.clip(a, 0, 1))
                    colors_dirty = True

                for trace in self.traces:
                    val = data.get(trace.varname)
                    if val is None:
                        continue
                    if trace.seg_idx is not None and hasattr(val, '__len__'):
                        trace.t.append(t)
                        trace.v.append(val[trace.seg_idx])
                    else:
                        trace.t.append(t)
                        trace.v.append(val)

            if colors_dirty:
                self.canvas3d.update()
            for trace in self.traces:
                if trace.t:
                    trace.plot_item.setData(np.asarray(trace.t), np.asarray(trace.v))
        except (EOFError, OSError):
            self.timer.stop()

    def _on_mouse_press(self, ev):
        self._mouse_start = ev.pos

    def _on_mouse_release(self, ev):
        if not self.has_morphology:
            return
        if not hasattr(self,'_mouse_start'):
            return
        dx = ev.pos[0] - self._mouse_start[0]
        dy = ev.pos[1] - self._mouse_start[1]
        del self._mouse_start
        if dx*dx + dy*dy > self.DRAG_THRESHOLD**2:
            return
        x,y = ev.pos; w,h=self.canvas3d.size; ps=self.canvas3d.pixel_scale
        xf, yf = int(x*ps), int((h-y-1)*ps)
        picked = self.mgr.pick(xf, yf, self.canvas3d)
        if picked:
            if not self.sim.handle_segment_click(*picked, self):
                self.select(*picked)

    def _resolve_seg_idx(self, sec, xloc):
        """Find the closest segment index for a given section and xloc."""
        if not self.has_morphology:
            return None
        diffs = np.abs(self.mgr.xlocs - xloc)
        mask  = np.array([n==sec for n in self.mgr.sec_names])[self.mgr.sec_idx]
        idxs  = np.where(mask)[0]
        if idxs.size:
            return idxs[np.argmin(diffs[idxs])]
        return None

    def select(self, sec, xloc):
        seg_idx = self._resolve_seg_idx(sec, xloc)
        if seg_idx is None:
            return
        self.selected = (sec, xloc)
        color = TRACE_COLORS[len(self.traces) % len(TRACE_COLORS)]
        label = f"{sec}@{xloc:.3f}"
        plot_item = self.plot2d.plot(pen=color, name=label)
        self.traces.append(_Trace(sec, xloc, seg_idx, plot_item, varname='v'))
        self.plot2d.setTitle(f"Voltage for {sec}@{xloc:.3f}")

    def add_scalar_trace(self, varname, label=None, color=None):
        """Add a trace for a scalar simulation variable (e.g. point process output)."""
        if label is None:
            label = varname
        if color is None:
            color = TRACE_COLORS[len(self.traces) % len(TRACE_COLORS)]
        plot_item = self.plot2d.plot(pen=color, name=label)
        self.traces.append(_Trace(sec=None, xloc=None, seg_idx=None,
                                  plot_item=plot_item, varname=varname))
        # enable y auto-range for scalar traces (0-1 range would be invisible in voltage range)
        vb = self.plot2d.getPlotItem().getViewBox()
        vb.enableAutoRange(y=True)
        vb.setLimits(yMin=None, yMax=None)

    def clear_traces(self):
        for trace in self.traces:
            self.plot2d.removeItem(trace.plot_item)
        self.traces.clear()
        self.selected = None
        legend = self.plot2d.getPlotItem().legend
        if legend:
            legend.clear()
        # reapply fixed y-range after clearing
        vb = self.plot2d.getPlotItem().getViewBox()
        vb.setLimits(yMin=self._vb_ymin, yMax=self._vb_ymax)
        vb.setRange(yRange=(self._vb_ymin, self._vb_ymax), padding=0)

    def keyPressEvent(self, ev):
        if self.sim.handle_key_press(ev.key(), self):
            return
        if ev.key() == Qt.Key.Key_Space:
            self.cmd_parent.send("reset")
            while self.data_parent.poll():
                self.data_parent.recv()
            for trace in self.traces:
                trace.t.clear()
                trace.v.clear()
                trace.plot_item.setData([], [])
            self.sim.handle_viewer_reset(self)
        elif ev.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.clear_traces()
            vb = self.plot2d.getPlotItem().getViewBox()
            vb.enableAutoRange(x=True, y=False)
        super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev):
        if self.sim.handle_key_release(ev.key(), self):
            return
        super().keyReleaseEvent(ev)

    def closeEvent(self, ev):
        if self.worker.is_alive():
            self.worker.terminate(); self.worker.join()
        self.timer.stop()
        super().closeEvent(ev)


MorphologyViewer = SimulationViewer


def run_visualizer(sim: Simulation, setup_callback=None):
    # only launch GUI in the original process, not in any spawned children
    if mp.current_process().name != "MainProcess":
        return

    # Windows multiprocessing setup
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)

    # now safe to create the QApplication and window
    app = QtWidgets.QApplication(sys.argv)
    w   = SimulationViewer(sim)
    if setup_callback is not None:
        setup_callback(w)
    w.show()
    vispy_app.run()
