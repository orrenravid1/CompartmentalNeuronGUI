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
        self.trace_t, self.trace_v = [], []
        self.trace = self.plot2d.plot(pen='b')
        vb = self.plot2d.getPlotItem().getViewBox()
        vb.setRange(yRange=(-80,50), padding=0)
        vb.enableAutoRange(x=True, y=False)
        vb.setLimits(yMin=-80, yMax=50)

        # layout
        w = QtWidgets.QWidget()
        hb = QtWidgets.QHBoxLayout(w)
        hb.addWidget(self.canvas3d.native)
        hb.addWidget(self.plot2d)
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
        self.timer.start(1000 // 200)

        self.selected = None
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

                if self.selected:
                    sec, xloc = self.selected
                    # find closest
                    diffs = np.abs(self.mgr.xlocs - xloc)
                    mask  = np.array([n==sec for n in self.mgr.sec_names])[self.mgr.sec_idx]
                    idxs  = np.where(mask)[0]
                    if idxs.size:
                        best = idxs[np.argmin(diffs[idxs])]
                        self.trace_t.append(t)
                        self.trace_v.append(arr[best])
                        if len(self.trace_t)>1000:
                            self.trace_t.pop(0); self.trace_v.pop(0)
                        self.trace.setData(self.trace_t, self.trace_v)
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
            self.select(*picked)

    def select(self, sec, xloc):
        self.selected = (sec, xloc)
        self.trace_t.clear(); self.trace_v.clear()
        self.trace.clear()
        self.plot2d.setTitle(f"Voltage for {sec}@{xloc:.3f}")

    def keyPressEvent(self, ev):
        if ev.key()==Qt.Key.Key_Space and self.geom_ready:
            self.cmd_parent.send("reset")
            while self.data_parent.poll():
                self.data_parent.recv()
            self.trace_t.clear(); self.trace_v.clear()
            self.trace.clear()
            vb = self.plot2d.getPlotItem().getViewBox()
            vb.enableAutoRange(x=True, y=False)
        super().keyPressEvent(ev)

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
