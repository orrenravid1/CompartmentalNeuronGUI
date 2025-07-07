#!/usr/bin/env python3
import sys
import os
import numpy as np
import multiprocessing as mp
from multiprocessing import Pipe, Process

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


def build_morphology_meta(secs):
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
    return meta


def load_swc_model(swc_path):
    """Import an SWC file into NEURON and return a list of Sections."""
    from neuron import h
    h.load_file("stdlib.hoc")
    h.load_file("import3d.hoc")
    r = h.Import3d_SWC_read()
    r.input(swc_path)
    gui = h.Import3d_GUI(r, False)
    gui.instantiate(None)
    return list(h.allsec())


class MorphologyManager:
    """Handles all VisPy instancing, picking and color‐mapping."""
    def __init__(self, view):
        self.view = view
        self.meta = None
        self.collection = None
        self.id_colors = None
        self.id_colors_caps = None
        self.orig_side = None
        self.orig_cap = None

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
            cylinder_segments=32, disk_slices=32,
            parent=self.view.scene
        )
        # disable lighting so we can override colors
        self.collection._side_mesh.shading = None
        self.collection._cap_mesh.shading  = None

        # picking colors
        self.id_colors      = np.stack([self.make_id_color(i) for i in range(N)], axis=0)
        self.id_colors_caps = np.vstack([self.id_colors, self.id_colors])

    def pick(self, x_fb, y_fb, canvas):
        # 1) swap in the ID‐colors
        side = self.collection._side_mesh
        cap  = self.collection._cap_mesh
        old_side = side.instance_colors
        old_cap  = cap.instance_colors
        side.instance_colors = self.id_colors
        cap.instance_colors  = self.id_colors_caps

        # 2) do an OFF‐SCREEN render (this does *not* modify the displayed canvas)
        img = canvas.render(region=(x_fb, y_fb, 1,1),
                            size=(1,1),
                            alpha=False)

        # 3) restore the real colors (still off‐screen—no update())
        side.instance_colors = old_side
        cap.instance_colors  = old_cap

        # 4) decode your pick ID from img:
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
        """Apply map_fn to data array, build RGBA buffer, and set colors."""
        norm = map_fn(data)
        new_cols = np.zeros((len(data),4), dtype=np.float32)
        new_cols[:,0] = norm
        new_cols[:,1] = 0.2
        new_cols[:,2] = 1.0 - norm
        new_cols[:,3] = 1.0
        self.collection.set_colors(new_cols)


def neuron_process(data_pipe, cmd_pipe, swc_path):
    """Worker process that loads morphology, simulates, and streams (t, v)."""
    from neuron import h

    secs = load_swc_model(swc_path)
    print("Num secs = ", len(secs))
    for sec in secs:
        sec.insert("hh" if "dendrite" not in sec.name() else "pas")
        if 'soma' not in sec.name():
            sec.nseg = 10
    meta = build_morphology_meta(secs)

    name2sec = {sec.name(): sec for sec in secs}
    refs = [(name2sec[m['sec_name']], m['xloc']) for m in meta]
    N = len(refs)

    h.dt = 0.2
    vt = h.Vector(); vt.record(h._ref_t)
    vs = []
    for sec,xloc in refs:
        v = h.Vector(); v.record(sec(xloc)._ref_v)
        vs.append(v)

    soma = next(sec for sec in secs if 'soma' in sec.name().lower())
    iclamps = []
    for d,du,a in [(2,10,1),(20,10,1),(40,10,1),(60,20,1)]:
        icl=h.IClamp(soma(0.5)); icl.delay, icl.dur, icl.amp = d,du,a
        iclamps.append(icl)

    h.finitialize(-65.0)
    try:
        while True:
            h.fadvance()
            # reset?
            while cmd_pipe.poll():
                if cmd_pipe.recv()=="reset":
                    h.finitialize(-65.0)
            t = float(vt[-1])
            arr = np.fromiter((v[-1] for v in vs), np.float32, count=N)
            data_pipe.send((t, arr))
    finally:
        data_pipe.close(); cmd_pipe.close()


class MorphologyViewer(QtWidgets.QMainWindow):
    def __init__(self, swc_path):
        super().__init__()
        self.setWindowTitle("Morphology + Voltage")

        # VisPy 3D scene setup
        self.canvas3d = scene.SceneCanvas(keys='interactive',
                                          bgcolor='white', show=False)
        self.view     = self.canvas3d.central_widget.add_view()
        self.view.camera = TurntableCamera(fov=60, distance=200,
                                           elevation=30, azimuth=30,
                                           translate_speed=100, up='+z')

        # manager instantiation
        self.morph_mgr = MorphologyManager(self.view)

        # load once in parent, set up geometry
        secs = load_swc_model(swc_path)
        meta = build_morphology_meta(secs)
        self.morph_mgr.set_morphology(meta)

        # selection state
        self.DRAG_THRESHOLD = 5
        self.canvas3d.events.mouse_press .connect(self._on_mouse_press)
        self.canvas3d.events.mouse_release.connect(self._on_mouse_release)

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
        w  = QtWidgets.QWidget()
        hb = QtWidgets.QHBoxLayout(w)
        hb.addWidget(self.canvas3d.native)
        hb.addWidget(self.plot2d)
        self.setCentralWidget(w)
        self.resize(1200,600)

        self.select(meta[0]['sec_name'], 0.5)

        # multiprocessing setup
        self.data_parent, data_child = Pipe()
        self.cmd_parent,  cmd_child  = Pipe()
        self.worker = Process(
            target=neuron_process,
            args=(data_child, cmd_child, swc_path),
            daemon=True
        )
        self._child_pipes = (data_child, cmd_child)
        QtCore.QTimer.singleShot(0, self._start_worker)

        # polling timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(1000//30)

    def _start_worker(self):
        self.worker.start()
        data_child, cmd_child = self._child_pipes
        data_child.close(); cmd_child.close()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Space:
            # 1) send reset
            self.cmd_parent.send("reset")

            # 2) flush any queued data so the next sample is truly from t=0
            while self.data_parent.poll():
                self.data_parent.recv()

            # 3) clear your buffers & curve
            self.trace_t.clear()
            self.trace_v.clear()
            self.plot2d.clear()
            self.trace = self.plot2d.plot(pen='b')
            vb = self.plot2d.getPlotItem().getViewBox()
            vb.enableAutoRange(x=True, y=False)


    def _on_mouse_press(self, ev):
        self._mouse_start = ev.pos

    def _on_mouse_release(self, ev):
        if getattr(self, '_mouse_start', None) is None:
            return
        dx = ev.pos[0]-self._mouse_start[0]
        dy = ev.pos[1]-self._mouse_start[1]
        self._mouse_start = None
        if dx*dx+dy*dy > self.DRAG_THRESHOLD**2:
            return

        # compute fb coords
        x,y = ev.pos
        w,h = self.canvas3d.size
        ps  = self.canvas3d.pixel_scale
        xf = int(x*ps); yf = int((h-y-1)*ps)

        picked = self.morph_mgr.pick(xf, yf, self.canvas3d)
        if picked:
            sec_name, xloc = picked
            self.select(sec_name, xloc)
    
    def select(self, sec_name, xloc):
        self.selected = (sec_name, xloc)
        self.trace_t.clear()
        self.trace_v.clear()
        self.plot2d.setTitle(f"Voltage for {sec_name}@{xloc:.3f}")

    def _poll(self):
        try:
            last = None
            while self.data_parent.poll():
                last = self.data_parent.recv()
            if last:
                t, v = last
                # update 3D colors
                self.morph_mgr.update_colors(v, lambda arr: np.clip((arr+80)/130,0,1))

                if self.selected:
                    sec_name, sel_xloc = self.selected
                    # find all indices in that section
                    candidates = [
                        (i, abs(m['xloc'] - sel_xloc))
                        for i, m in enumerate(self.morph_mgr.meta)
                        if m['sec_name'] == sec_name
                    ]
                    if candidates:
                        if self.trace_t and t < self.trace_t[-1]:
                            self.trace_t.clear()
                            self.trace_v.clear()
                        # pick the one with minimal |xloc - sel_xloc|
                        idx = min(candidates, key=lambda x: x[1])[0]
                        self.trace_t.append(t)
                        self.trace_v.append(v[idx])
                        if len(self.trace_t) > 1000:
                            self.trace_t.pop(0)
                            self.trace_v.pop(0)
                        self.trace.setData(self.trace_t, self.trace_v)
                        self.plot2d.update()
        except (EOFError, OSError):
            self.timer.stop()

    def closeEvent(self, ev):
        if self.worker.is_alive():
            self.worker.terminate(); self.worker.join()
        self.timer.stop()
        super().closeEvent(ev)


if __name__=="__main__":
    mp.set_start_method('spawn', force=True)
    app = QtWidgets.QApplication(sys.argv)
    viewer = MorphologyViewer(os.path.join("res","Animal_2_Basal_2.CNG.swc"))
    viewer.show()
    vispy_app.run()
