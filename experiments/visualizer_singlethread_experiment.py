#!/usr/bin/env python3
import sys
import os
import time
import numpy as np
import multiprocessing as mp
from multiprocessing import Pipe, Process, cpu_count, Pool

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import Qt
from vispy import use
# ─── VisPy + PyQt6 backend ────────────────────────────────────────────────
use(app='pyqt6', gl='gl+')
from vispy import scene, app as vispy_app
from vispy.scene.cameras import TurntableCamera
from scipy.spatial.transform import Rotation
import pyqtgraph as pg

# replace with your actual import path
from src.vispyutils.cappedcylindercollection import CappedCylinderCollection
from src.neuronutils.swc_utils import load_swc_neuron


def build_morphology_meta(secs):
    t0 = time.perf_counter()
    """Build per-cylinder metadata from a dict name→h.Section."""
    t_neurite = np.array([0.7,0.7,0.7,1.0], dtype=np.float32)
    name_to_sec = {sec.name(): sec for sec in secs}
    meta = []
    for sec_name, sec in name_to_sec.items():
        pts = np.array([[sec.x3d(i), sec.y3d(i), sec.z3d(i)]
                        for i in range(int(sec.n3d()))], dtype=np.float32)
        if len(pts) < 2:
            continue
        d   = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        cum = np.concatenate([[0.0], np.cumsum(d)])
        total = cum[-1] if cum[-1] > 0 else 1.0
        for i in range(len(pts)-1):
            p0, p1 = pts[i], pts[i+1]
            L = np.linalg.norm(p1 - p0)
            if L < 1e-6:
                continue
            mid = 0.5*(p0 + p1)
            rad = 0.5*(sec.diam3d(i) + sec.diam3d(i+1))
            z  = np.array([0,0,1], dtype=np.float32)
            dn = (p1 - p0) / L
            ax = np.cross(z, dn)
            if np.linalg.norm(ax) < 1e-6:
                R = np.eye(3, dtype=np.float32)
            else:
                ax /= np.linalg.norm(ax)
                ang = np.arccos(np.clip(np.dot(z, dn), -1,1))
                R = Rotation.from_rotvec(ax*ang).as_matrix().astype(np.float32)
            midlen = cum[i] + 0.5*d[i]
            xloc   = float(midlen / total)
            meta.append({
                'position':    mid,
                'radius':      rad,
                'length':      L,
                'orientation': R,
                'color':       t_neurite,
                'sec_name':    sec_name,
                'xloc':        xloc
            })
    elapsed = time.perf_counter() - t0
    print(f"Built meta in {elapsed:.3f} s")
    return meta


class MorphologyManager:
    """Handles all VisPy instancing, picking and color‐mapping."""
    def __init__(self, view):
        self.view = view
        self.meta = None
        self.collection = None
        self.id_colors = None
        self.id_colors_caps = None

    @staticmethod
    def make_id_color(i):
        cid = i+1
        return np.array([
            (cid        & 0xFF)/255.0,
            ((cid >> 8) & 0xFF)/255.0,
            ((cid >> 16)& 0xFF)/255.0,
            1.0
        ], dtype=np.float32)

    def set_morphology(self, meta):
        self.meta = meta
        N = len(meta)
        pos  = np.stack([m['position']    for m in meta], axis=0)
        rad  = np.array([m['radius']      for m in meta], dtype=np.float32)
        ht   = np.array([m['length']      for m in meta], dtype=np.float32)
        ori  = np.stack([m['orientation'] for m in meta], axis=0)
        cols = np.stack([m['color']       for m in meta], axis=0)

        self.collection = CappedCylinderCollection(
            positions=pos, radii=rad, heights=ht,
            orientations=ori, colors=cols,
            cylinder_segments=16, disk_slices=16,
            parent=self.view.scene
        )
        # disable lighting so we can override colors
        self.collection._side_mesh.shading = None
        self.collection._cap_mesh.shading  = None

        # prepare picking buffers
        self.id_colors      = np.stack([self.make_id_color(i) for i in range(N)], axis=0)
        self.id_colors_caps = np.vstack([self.id_colors, self.id_colors])

    def pick(self, x_fb, y_fb, canvas):
        side = self.collection._side_mesh
        cap  = self.collection._cap_mesh
        old_side = side.instance_colors
        old_cap  = cap.instance_colors
        # 1) swap in ID‐colors
        side.instance_colors = self.id_colors
        cap.instance_colors  = self.id_colors_caps
        # 2) off‐screen render
        img = canvas.render(region=(x_fb, y_fb, 1,1),
                            size=(1,1), alpha=False)
        # 3) restore
        side.instance_colors = old_side
        cap.instance_colors  = old_cap

        pix = img[0,0]
        if pix.dtype != np.uint8:
            pix = np.round(pix*255).astype(int)
        cid = int(pix[0]) | (int(pix[1])<<8) | (int(pix[2])<<16)
        idx = cid-1 if cid>0 else None
        if idx is None or not (0 <= idx < len(self.meta)):
            return None
        m = self.meta[idx]
        return m['sec_name'], m['xloc']

    def update_colors(self, data, map_fn):
        norm = map_fn(data)
        new_cols = np.zeros((len(data),4), dtype=np.float32)
        new_cols[:,0] = norm
        new_cols[:,1] = 0.2
        new_cols[:,2] = 1.0 - norm
        new_cols[:,3] = 1.0
        self.collection.set_colors(new_cols)


def neuron_process(data_pipe, cmd_pipe, swc_path):
    """Worker process: loads morphology, sends meta, then simulates & streams."""
    from neuron import h

    print("Loading swc model...")
    # load & insert
    secs = load_swc_neuron(swc_path)
    print ("Loaded swc model.")
    for sec in secs:
        sec.insert("hh" if "dendrite" not in sec.name() else "pas")
        if 'soma' not in sec.name():
            sec.nseg = 10

    print("Building morphology meta...")
    # build & send morphology once
    meta = build_morphology_meta(secs)
    print("Built morpholog meta.")
    print("Sending morphology meta...")
    data_pipe.send(meta)
    print("Sent morphology meta.")

    # set up recording
    name2sec = {sec.name(): sec for sec in secs}
    refs = [(name2sec[m['sec_name']], m['xloc']) for m in meta]
    N = len(refs)

    h.dt = 0.2
    vt = h.Vector(); vt.record(h._ref_t)
    pvs = h.PtrVector(N)
    vs  = h.Vector(N)
    for i, (sec, xloc) in enumerate(refs):
        pvs.pset(i, sec(xloc)._ref_v)

    # IClamps
    soma = next(sec for sec in secs if 'soma' in sec.name().lower())
    iclamps = []
    for d, du, a in [(2,10,1),(20,10,1),(40,10,1),(60,20,1)]:
        icl = h.IClamp(soma(0.5))
        icl.delay, icl.dur, icl.amp = d, du, a
        iclamps.append(icl)

    h.finitialize(-65.0)
    try:
        while True:
            h.fadvance()
            # handle reset
            while cmd_pipe.poll():
                if cmd_pipe.recv() == "reset":
                    h.finitialize(-65.0)
            # send one frame
            t = float(vt[-1])
            pvs.gather(vs)
            arr = vs.as_numpy()
            data_pipe.send((t, arr))
    finally:
        data_pipe.close()
        cmd_pipe.close()


class MorphologyViewer(QtWidgets.QMainWindow):
    def __init__(self, swc_path):
        super().__init__()
        self.setWindowTitle("Morphology + Voltage")

        # VisPy 3D scene
        self.canvas3d = scene.SceneCanvas(keys='interactive',
                                          bgcolor='white', show=False)
        self.view     = self.canvas3d.central_widget.add_view()
        self.view.camera = TurntableCamera(fov=60, distance=200,
                                           elevation=30, azimuth=30,
                                           translate_speed=100, up='+z')

        # geometry manager
        self.mgr = MorphologyManager(self.view)
        self.geom_ready = False

        # PyQtGraph 2D plot (white bg, blue trace)
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
        w  = QtWidgets.QWidget()
        hb = QtWidgets.QHBoxLayout(w)
        hb.addWidget(self.canvas3d.native)
        hb.addWidget(self.plot2d)
        self.setCentralWidget(w)
        self.resize(1200,600)

        # duplex pipes
        self.data_parent, self.data_child = Pipe(duplex=True)
        self.cmd_parent,  self.cmd_child  = Pipe(duplex=True)

        # spawn worker
        self.worker = Process(
            target=neuron_process,
            args=(self.data_child, self.cmd_child, swc_path)
        )
        QtCore.QTimer.singleShot(0, self._start_worker)

        # polling timer (~30 Hz)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(1000 // 30)

        # selection
        self.selected = None
        self.DRAG_THRESHOLD = 5

    def _start_worker(self):
        self.worker.start()
        # close child‐ends in parent
        self.data_child.close()
        self.cmd_child.close()

    def _poll(self):
        try:
            while self.data_parent.poll():
                msg = self.data_parent.recv()
                if not self.geom_ready:

                    # first message: morphology meta
                    meta = msg
                    self.mgr.set_morphology(meta)
                    self.geom_ready = True
                    # enable picking
                    self.canvas3d.events.mouse_press .connect(self._on_mouse_press)
                    self.canvas3d.events.mouse_release.connect(self._on_mouse_release)
                    # initial select
                    sec0, x0 = meta[0]['sec_name'], 0.5
                    self.select(sec0, x0)
                    continue

                # subsequent messages: (t, arr)
                t, arr = msg
                # update 3D
                self.mgr.update_colors(arr, lambda a: np.clip((a+80)/130, 0, 1))
                self.canvas3d.update()
                # update 2D
                if self.selected:
                    sec, xloc = self.selected
                    cand = [
                        (i, abs(m['xloc'] - xloc))
                        for i, m in enumerate(self.mgr.meta)
                        if m['sec_name'] == sec
                    ]
                    if cand:
                        idx = min(cand, key=lambda x: x[1])[0]
                        self.trace_t.append(t)
                        self.trace_v.append(arr[idx])
                        if len(self.trace_t) > 1000:
                            self.trace_t.pop(0)
                            self.trace_v.pop(0)
                        self.trace.setData(self.trace_t, self.trace_v)
        except (EOFError, OSError):
            self.timer.stop()

    def _on_mouse_press(self, ev):
        self._mouse_start = ev.pos

    def _on_mouse_release(self, ev):
        if not hasattr(self, '_mouse_start'):
            return
        dx = ev.pos[0] - self._mouse_start[0]
        dy = ev.pos[1] - self._mouse_start[1]
        del self._mouse_start
        if dx*dx + dy*dy > self.DRAG_THRESHOLD**2:
            return
        x, y = ev.pos
        w, h = self.canvas3d.size
        ps   = self.canvas3d.pixel_scale
        xf, yf = int(x*ps), int((h-y-1)*ps)
        picked = self.mgr.pick(xf, yf, self.canvas3d)
        if picked:
            self.select(*picked)

    def select(self, sec, xloc):
        self.selected = (sec, xloc)
        self.trace_t.clear()
        self.trace_v.clear()
        self.trace.clear()
        self.plot2d.setTitle(f"Voltage for {sec}@{xloc:.3f}")

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Space and self.geom_ready:
            # reset worker simulation
            self.cmd_parent.send("reset")
            # flush any queued frames
            while self.data_parent.poll():
                self.data_parent.recv()
            # clear 2D trace
            self.trace_t.clear()
            self.trace_v.clear()
            self.trace.clear()
            vb = self.plot2d.getPlotItem().getViewBox()
            vb.enableAutoRange(x=True, y=False)
        super().keyPressEvent(ev)

    def closeEvent(self, ev):
        if self.worker.is_alive():
            self.worker.terminate()
            self.worker.join()
        self.timer.stop()
        super().closeEvent(ev)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    swc_path = os.path.join("res", "Animal_2_Basal_2.CNG.swc")
    app = QtWidgets.QApplication(sys.argv)
    w = MorphologyViewer(swc_path)
    w.show()
    vispy_app.run()
