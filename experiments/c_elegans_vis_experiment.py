#!/usr/bin/env python3
import sys
import os
import time
import numpy as np
import multiprocessing as mp
from multiprocessing import Pipe, Process

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import Qt
from vispy import use
use(app='pyqt6', gl='gl+')
from vispy import scene, app as vispy_app
from vispy.scene.cameras import TurntableCamera
import pyqtgraph as pg

from src.vispyutils.cappedcylindercollection import CappedCylinderCollection
from src.neuronutils.swc_utils import load_swc_multi


def build_morphology_meta(secs):
    """
    Pure‐NumPy, vectorized builder for per‐segment metadata.
    Filters out zero‐length segments to avoid divide‐by‐zero.
    """
    import time, numpy as np

    t0 = time.perf_counter()

    # ─── Gather per‐section data ─────────────────────────────────────────────
    sec_names = []
    P0_list, P1_list = [], []
    D0_list, D1_list = [], []
    CUM_list, TOT_list, S_list = [], [], []

    for si, sec in enumerate(secs):
        n3d = int(sec.n3d())
        if n3d < 2:
            continue
        sec_names.append(sec.name())

        pts   = np.stack([[sec.x3d(i), sec.y3d(i), sec.z3d(i)]
                          for i in range(n3d)], axis=0).astype(np.float32)
        diams = np.array([sec.diam3d(i) for i in range(n3d)],
                         dtype=np.float32)

        diffs = pts[1:] - pts[:-1]                  # (n3d-1,3)
        dlen  = np.linalg.norm(diffs, axis=1)       # (n3d-1,)
        cum   = np.concatenate(([0.0], np.cumsum(dlen)))[:-1]
        total = cum[-1] + dlen[-1] if dlen.sum() > 0 else 1.0

        P0_list.append(pts[:-1])
        P1_list.append(pts[1:])
        D0_list.append(diams[:-1])
        D1_list.append(diams[1:])
        CUM_list.append(cum)
        TOT_list.append(np.full_like(dlen, total, dtype=np.float32))
        S_list.append(np.full_like(dlen, si, dtype=np.int32))

    # ─── Stack into flat arrays ──────────────────────────────────────────────
    P0  = np.vstack(P0_list)       # (M,3)
    P1  = np.vstack(P1_list)
    D0  = np.concatenate(D0_list)  # (M,)
    D1  = np.concatenate(D1_list)
    CUM = np.concatenate(CUM_list)
    TOT = np.concatenate(TOT_list)
    S   = np.concatenate(S_list)
    M   = P0.shape[0]

    # ─── Compute raw lengths and mask out zeros ──────────────────────────────
    raw_L = np.linalg.norm(P1 - P0, axis=1)         # (M,)
    mask  = raw_L > 1e-6
    P0, P1, D0, D1 = P0[mask], P1[mask], D0[mask], D1[mask]
    CUM, TOT, S   = CUM[mask], TOT[mask], S[mask]
    L             = raw_L[mask]

    # ─── Midpoints, xloc, radii, colors ─────────────────────────────────────
    mid   = 0.5*(P0 + P1)                            # (K,3)
    xloc  = (CUM + 0.5*L) / TOT                      # (K,)
    rad   = 0.5*(D0 + D1)                            # (K,)
    col   = np.tile([0.7,0.7,0.7,1.0], (len(L),1)).astype(np.float32)

    # ─── Bulk Rodrigues rotations ────────────────────────────────────────────
    dn    = (P1 - P0) / L[:,None]                    # (K,3)
    cos_t = dn[:,2]
    ang   = np.arccos(np.clip(cos_t, -1.0, 1.0))
    ax    = np.cross(np.repeat([[0,0,1]], len(L), 0), dn)
    ax_n  = np.linalg.norm(ax, axis=1, keepdims=True)
    ax_u  = ax / np.where(ax_n>1e-6, ax_n, 1.0)
    ux, uy, uz = ax_u.T

    K    = np.zeros((len(L),3,3), dtype=np.float32)
    K[:,0,1] = -uz; K[:,0,2] =  uy
    K[:,1,0] =  uz; K[:,1,2] = -ux
    K[:,2,0] = -uy; K[:,2,1] =  ux
    K2   = K @ K

    sin_t = np.sin(ang)[:,None,None]
    one_c = (1.0 - cos_t)[:,None,None]
    I     = np.eye(3, dtype=np.float32)[None,:,:]

    R = I + sin_t*K + one_c*K2                    # (K,3,3)

    elapsed = time.perf_counter() - t0
    print(f"Meta file generated in {elapsed:.2f}s (filtered {M-len(L)} zero-length segments)")

    return {
        'positions':    mid,
        'orientations':R,
        'radii':        rad,
        'lengths':      L,
        'colors':       col,
        'sec_names':    sec_names,
        'sec_idx':      S,
        'xloc':         xloc
    }


def neuron_process(data_pipe, cmd_pipe):
    """Worker process: send geometry meta, then simulate & stream (t, v)."""
    from neuron import h

    t0 = time.perf_counter()
    
    swc_path = os.path.join("res","celegans_cells_swc")
    swc_files = [f for f in os.listdir(swc_path)]
    secs = []
    for i,swcf in enumerate(swc_files):
        print(f"Loading cell {swcf}")
        cell_name = swcf.split('.')[0]
        trees = load_swc_multi(os.path.join(swc_path, swcf), cell_name)
        seclists = trees.values()
        for seclist in seclists:
            for sec in seclist:
                secs.append(sec)

    elapsed = time.perf_counter() - t0
    print(f"SWC Loaded in {elapsed:.2f}s")

    for sec in secs:
        sec.insert("hh" if "dendrite" not in sec.name() else "pas")
        if "soma" not in sec.name():
            sec.nseg = 10

    # build & send morphology meta
    meta = build_morphology_meta(secs)
    data_pipe.send(meta)

    # recordings
    name2sec = {sec.name(): sec for sec in secs}
    refs = [(name2sec[meta['sec_names'][meta['sec_idx'][i]]], meta['xloc'][i])
            for i in range(len(meta['sec_idx']))]
    N = len(refs)

    h.dt = 0.1
    vt = h.Vector()
    vt.record(h._ref_t)
    pvs = h.PtrVector(N)
    vs  = h.Vector(N)
    for i, (sec, x) in enumerate(refs):
        pvs.pset(i, sec(x)._ref_v)

    somas = [sec for sec in secs if 'soma' in sec.name().lower()]
    iclamps = []
    for soma in somas:
        for d,du,a in [(2,5,0.2),(20,5,0.2),(40,5,0.2),(60,5,0.2),(80,5,0.2)]:
            icl = h.IClamp(soma(0.5))
            icl.delay, icl.dur, icl.amp = d, du, a
            iclamps.append(icl)

    h.finitialize(-65.0)
    try:
        while True:
            h.fadvance()
            while cmd_pipe.poll():
                if cmd_pipe.recv()=="reset":
                    h.finitialize(-65.0)
            t   = float(vt[-1])
            pvs.gather(vs)
            arr = vs.as_numpy()
            data_pipe.send((t, arr))
    finally:
        data_pipe.close()
        cmd_pipe.close()


class MorphologyManager:
    """VisPy instancing, picking & color‐mapping from flat arrays."""
    def __init__(self, view):
        self.view = view

    def set_morphology(self, meta):
        
        t0 = time.perf_counter()

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
    def __init__(self):
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
            target=neuron_process,
            args=(self.data_child, self.cmd_child)
        )
        QtCore.QTimer.singleShot(0, self._start_worker)

        # polling
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(1000 // 60)

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
                        if self.trace_t and t < self.trace_t[-1]:
                            self.trace_t.clear()
                            self.trace_v.clear()
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


if __name__=="__main__":
    mp.set_start_method('spawn', force=True)
    app = QtWidgets.QApplication(sys.argv)
    w = MorphologyViewer()
    w.show()
    vispy_app.run()
