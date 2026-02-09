import time
import multiprocessing as mp
from multiprocessing import Pipe, Process
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

class _Trace:
    """A single recorded trace on the plot."""
    __slots__ = ('sec', 'xloc', 'seg_idx', 'plot_item', 't', 'v')

    def __init__(self, sec, xloc, seg_idx, plot_item):
        self.sec = sec
        self.xloc = xloc
        self.seg_idx = seg_idx
        self.plot_item = plot_item
        self.t = []
        self.v = []


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
        cols = np.zeros((len(data),4), dtype=np.float32)
        cols[:,0] = norm
        cols[:,1] = 0.2
        cols[:,2] = 1.0 - norm
        cols[:,3] = 1.0
        self.collection.set_colors(cols)


class MorphologyViewer(QtWidgets.QMainWindow):
    def __init__(self, sim: Simulation):
        super().__init__()
        self.setWindowTitle("Morphology + Voltage")
        self.statusBar().showMessage("Loading morphology…")

        # 3D canvas
        self.canvas3d = scene.SceneCanvas(keys='interactive',
                                          bgcolor='white', show=False)
        self.view     = self.canvas3d.central_widget.add_view()
        self.view.camera = TurntableCamera(fov=60, distance=200,
                                           elevation=30, azimuth=30,
                                           translate_speed=100, up='+z')

        self.mgr        = MorphologyManager(self.view)
        self.geom_ready = False

        # 2D plot
        self.plot2d = pg.PlotWidget(title="Voltage for segment")
        self.plot2d.setLabel('bottom','Time','ms')
        self.plot2d.setLabel('left','Voltage','mV')
        self.plot2d.setBackground('w')
        self.plot2d.addLegend(offset=(10, 10))
        self.traces: list[_Trace] = []
        vb = self.plot2d.getPlotItem().getViewBox()
        # fixed y-range for voltage, allow x auto-range
        self._vb_ymin, self._vb_ymax = -80, 80
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
        except Exception:
            specs = {}

        # If simulation exposes nothing but has a dt attribute, provide a fallback
        if not specs and hasattr(sim, 'dt'):
            specs = {
                'dt': {'type': 'float', 'min': 0.01, 'max': 1.0, 'steps': 100, 'default': getattr(sim, 'dt', 0.1), 'label': 'dt (ms)'}
            }

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
                init = spec.get('default', getattr(sim, name, mn))
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
        self._assign_iclamp_mode = False
        self._iclamp_target = None
        self.DRAG_THRESHOLD = 5

    def _start_worker(self):
        self._load_t0 = time.perf_counter()
        self.worker.start()
        self.data_child.close()
        self.cmd_child.close()

    def _poll(self):
        try:
            while self.data_parent.poll():
                msg = self.data_parent.recv()

                if not self.geom_ready:
                    # first message is meta dict
                    self.mgr.set_morphology(msg)
                    elapsed = time.perf_counter() - self._load_t0
                    self.statusBar().showMessage(f"Loaded in {elapsed:.2f}s")
                    print(f"Loaded in {elapsed:.2f}s")
                    QtCore.QTimer.singleShot(3000, self.statusBar().clearMessage)

                    self.geom_ready = True
                    self.canvas3d.events.mouse_press .connect(self._on_mouse_press)
                    self.canvas3d.events.mouse_release.connect(self._on_mouse_release)
                    # auto‐select first section
                    self.select(self.mgr.sec_names[0], 0.5)
                    continue

                # subsequent messages: (t, voltages)
                ## TODO: More generic data consumption
                t, arr = msg
                self.mgr.update_colors(arr, lambda a: np.clip((a+80)/130,0,1))
                self.canvas3d.update()

                for trace in self.traces:
                    trace.t.append(t)
                    trace.v.append(arr[trace.seg_idx])
                    if len(trace.t) > 1000:
                        trace.t.pop(0); trace.v.pop(0)
                    trace.plot_item.setData(trace.t, trace.v)
        except (EOFError, OSError):
            self.timer.stop()

    def _on_mouse_press(self, ev):
        self._mouse_start = ev.pos

    def _on_mouse_release(self, ev):
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
            # If user is holding the '1' key, assign IClamp target instead
            if getattr(self, '_assign_iclamp_mode', False):
                sec, xloc = picked
                self._iclamp_target = (sec, xloc)
                self.statusBar().showMessage(f"IClamp target set to {sec}@{xloc:.3f}")
                QtCore.QTimer.singleShot(3000, self.statusBar().clearMessage)
            else:
                self.select(*picked)

    def _resolve_seg_idx(self, sec, xloc):
        """Find the closest segment index for a given section and xloc."""
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
        self.traces.append(_Trace(sec, xloc, seg_idx, plot_item))
        self.plot2d.setTitle(f"Voltage for {sec}@{xloc:.3f}")

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
        if not self.geom_ready:
            return super().keyPressEvent(ev)
        # Hold '1' to assign IClamp location on next click
        if ev.key() == Qt.Key.Key_1:
            self._assign_iclamp_mode = True
            self.statusBar().showMessage("IClamp-assign mode: click a segment to set target")
            return
        if ev.key() == Qt.Key.Key_Space:
            self.cmd_parent.send("reset")
            while self.data_parent.poll():
                self.data_parent.recv()
            ##self.clear_traces()
            vb = self.plot2d.getPlotItem().getViewBox()
            vb.enableAutoRange(x=True, y=False)
            # reapply fixed y-range (ensure not auto-ranging)
            vb.setLimits(yMin=self._vb_ymin, yMax=self._vb_ymax)
            vb.setRange(yRange=(self._vb_ymin, self._vb_ymax), padding=0)
        elif ev.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.clear_traces()
            vb = self.plot2d.getPlotItem().getViewBox()
            vb.enableAutoRange(x=True, y=False)
        elif ev.key() == Qt.Key.Key_I:
            # apply an IClamp at the currently selected location (or first section)
            if not self.geom_ready:
                return
            if getattr(self, '_iclamp_target', None) is not None:
                sec_name, xloc = self._iclamp_target
            elif self.selected is not None:
                sec_name, xloc = self.selected
            else:
                sec_name = self.mgr.sec_names[0]
                xloc = 0.5
            payload = {'sec_name': sec_name, 'xloc': xloc, 'dur': 2.0, 'amp': 0.3}
            try:
                self.cmd_parent.send(("action", "iclamp", payload))
            except Exception:
                pass
        super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev):
        # release '1' to exit assign mode
        if getattr(self, '_assign_iclamp_mode', False) and ev.key() == Qt.Key.Key_1:
            self._assign_iclamp_mode = False
            self.statusBar().clearMessage()
            return
        super().keyReleaseEvent(ev)

    def closeEvent(self, ev):
        if self.worker.is_alive():
            self.worker.terminate(); self.worker.join()
        self.timer.stop()
        super().closeEvent(ev)

def run_visualizer(sim: Simulation):
    # only launch GUI in the original process, not in any spawned children
    if mp.current_process().name != "MainProcess":
        return

    # Windows multiprocessing setup
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)

    # now safe to create the QApplication and window
    app = QtWidgets.QApplication(sys.argv)
    w   = MorphologyViewer(sim)
    w.show()
    vispy_app.run()
