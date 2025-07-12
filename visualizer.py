#!/usr/bin/env python3
import sys
import os
import time
import numpy as np
import multiprocessing as mp
from multiprocessing import Pipe, Process, cpu_count
from multiprocessing.pool import Pool

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import Qt
from vispy import use
use(app='pyqt6', gl='gl+')
from vispy import scene, app as vispy_app
from vispy.scene.cameras import TurntableCamera
from scipy.spatial.transform import Rotation
import pyqtgraph as pg

from src.vispyutils.cappedcylindercollection import CappedCylinderCollection
from src.neuronutils.swc_utils import load_swc_model


def _process_chunk_flat(chunk, offset):
    """
    Compute segment metadata for one chunk of sections.
    Returns (pos, ori, rad, ht, col, sec_idx_arr, xloc_arr, offset).
    """
    t_neurite = np.array([0.7, 0.7, 0.7, 1.0], dtype=np.float32)

    # count total segments in this chunk
    local_len = sum(max(0, pts.shape[0] - 1) for _, pts, diams in chunk)
    pos      = np.zeros((local_len, 3),   np.float32)
    ori      = np.zeros((local_len, 3, 3), np.float32)
    rad      = np.zeros((local_len,),     np.float32)
    ht       = np.zeros((local_len,),     np.float32)
    col      = np.zeros((local_len, 4),   np.float32)
    sec_idxs = np.zeros((local_len,),     np.int32)
    xlocs    = np.zeros((local_len,),     np.float32)

    write_i = 0
    for sec_idx, pts, diams in chunk:
        npt = pts.shape[0]
        if npt < 2:
            continue
        d   = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        cum = np.concatenate([[0.0], np.cumsum(d)])
        total = cum[-1] if cum[-1] > 0 else 1.0

        for i in range(npt - 1):
            L = d[i]
            if L < 1e-6:
                continue
            p0, p1 = pts[i], pts[i+1]
            mid    = 0.5 * (p0 + p1)
            rad_v  = 0.5 * (diams[i] + diams[i+1])

            # orientation
            z  = np.array([0,0,1], np.float32)
            dn = (p1 - p0) / L
            ax = np.cross(z, dn)
            if np.linalg.norm(ax) < 1e-6:
                R = np.eye(3, dtype=np.float32)
            else:
                ax /= np.linalg.norm(ax)
                ang = np.arccos(np.clip(np.dot(z, dn), -1, 1))
                R   = Rotation.from_rotvec(ax*ang).as_matrix().astype(np.float32)

            midlen = cum[i] + 0.5 * d[i]
            xloc   = float(midlen / total)

            pos[write_i]      = mid
            ori[write_i]      = R
            rad[write_i]      = rad_v
            ht[write_i]       = L
            col[write_i]      = t_neurite
            sec_idxs[write_i] = sec_idx
            xlocs[write_i]    = xloc

            write_i += 1

    return pos, ori, rad, ht, col, sec_idxs, xlocs, offset


def build_morphology_meta(secs):
    t0 = time.perf_counter()
    """
    Build segment metadata in parallel with numpy buffer preallocation.
    Returns the flat arrays plus the section‐name list.
    """
    # 1) extract picklable data per section
    sec_names = [sec.name() for sec in secs]
    name2idx  = {n: i for i, n in enumerate(sec_names)}

    sec_data = []
    for sec in secs:
        npt = int(sec.n3d())
        if npt < 2:
            continue
        pts = np.stack([[sec.x3d(i), sec.y3d(i), sec.z3d(i)]
                        for i in range(npt)], axis=0).astype(np.float32)
        diams = np.array([sec.diam3d(i) for i in range(npt)], dtype=np.float32)
        sec_data.append((name2idx[sec.name()], pts, diams))

    # 2) compute total segments and per‐chunk offsets
    counts = [max(0, pts.shape[0] - 1) for _, pts, _ in sec_data]
    total = sum(counts)
    chunk_size = 100
    groups = [sec_data[i:i+chunk_size]
              for i in range(0, len(sec_data), chunk_size)]
    chunk_counts = [sum(counts[i*chunk_size:(i+1)*chunk_size])
                    for i in range(len(groups))]
    offsets = np.cumsum([0] + chunk_counts[:-1])

    # 3) allocate big arrays
    pos_g      = np.zeros((total, 3),   np.float32)
    ori_g      = np.zeros((total, 3, 3), np.float32)
    rad_g      = np.zeros((total,),     np.float32)
    ht_g       = np.zeros((total,),     np.float32)
    col_g      = np.zeros((total, 4),   np.float32)
    sec_idx_g  = np.zeros((total,),     np.int32)
    xlocs_g    = np.zeros((total,),     np.float32)

    # 4) parallel map
    args = [(groups[i], int(offsets[i])) for i in range(len(groups))]
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(_process_chunk_flat, args)

    # 5) merge chunk results
    for pos_c, ori_c, rad_c, ht_c, col_c, si_c, xl_c, off in results:
        L = pos_c.shape[0]
        pos_g[off:off+L]     = pos_c
        ori_g[off:off+L]     = ori_c
        rad_g[off:off+L]     = rad_c
        ht_g[off:off+L]      = ht_c
        col_g[off:off+L]     = col_c
        sec_idx_g[off:off+L] = si_c
        xlocs_g[off:off+L]   = xl_c
    
    elapsed = time.perf_counter() - t0
    print(f"Generated meta in {elapsed:.2f}s")

    return pos_g, ori_g, rad_g, ht_g, col_g, sec_names, sec_idx_g, xlocs_g


def neuron_process(data_pipe, cmd_pipe, swc_path):
    """Worker process: send geometry arrays then simulate/stream (t, v)."""
    from neuron import h

    secs = load_swc_model(swc_path)
    for sec in secs:
        sec.insert("hh" if "dendrite" not in sec.name() else "pas")
        if "soma" not in sec.name():
            sec.nseg = 10

    pos, ori, rad, ht, col, sec_names, sec_idx, xlocs = build_morphology_meta(secs)
    data_pipe.send((pos, ori, rad, ht, col, sec_names, sec_idx, xlocs))

    name2sec = {sec.name(): sec for sec in secs}
    refs = [(name2sec[sec_names[sec_idx[i]]], xlocs[i]) for i in range(len(sec_idx))]
    N = len(refs)

    h.dt = 0.2
    vt = h.Vector(); vt.record(h._ref_t)
    pvs = h.PtrVector(N)
    vs  = h.Vector(N)
    for i, (sec, xloc) in enumerate(refs):
        pvs.pset(i, sec(xloc)._ref_v)

    soma = next(sec for sec in secs if "soma" in sec.name().lower())
    iclamps = []
    for d, du, a in [(2,10,1),(20,10,1),(40,20,1),(60,20,1)]:
        icl = h.IClamp(soma(0.5))
        icl.delay, icl.dur, icl.amp = d, du, a
        iclamps.append(icl)

    h.finitialize(-65.0)
    try:
        while True:
            h.fadvance()
            while cmd_pipe.poll():
                if cmd_pipe.recv() == "reset":
                    h.finitialize(-65.0)
            t = float(vt[-1])
            pvs.gather(vs)
            arr = vs.as_numpy()
            data_pipe.send((t, arr))
    finally:
        data_pipe.close()
        cmd_pipe.close()


class MorphologyManager:
    """Handles VisPy instancing, picking & color‐mapping from flat arrays."""
    def __init__(self, view):
        self.view = view

    def set_morphology_arrays(self, pos, ori, rad, ht, col,
                              sec_names, sec_idx, xlocs):
        self.sec_names = sec_names
        self.sec_idx   = sec_idx
        self.xlocs     = xlocs

        self.collection = CappedCylinderCollection(
            positions=pos, radii=rad, heights=ht,
            orientations=ori, colors=col,
            cylinder_segments=32, disk_slices=32,
            parent=self.view.scene
        )
        self.collection._side_mesh.shading = None
        self.collection._cap_mesh.shading  = None

        N = pos.shape[0]
        def make_id_color(i):
            cid = i + 1
            return np.array([
                (cid        & 0xFF)/255.0,
                ((cid >>  8) & 0xFF)/255.0,
                ((cid >> 16) & 0xFF)/255.0,
                1.0
            ], dtype=np.float32)

        self.id_colors      = np.stack([make_id_color(i) for i in range(N)], axis=0)
        self.id_colors_caps = np.vstack([self.id_colors, self.id_colors])

    def pick(self, x_fb, y_fb, canvas):
        side, cap = self.collection._side_mesh, self.collection._cap_mesh
        old_side, old_cap = side.instance_colors, cap.instance_colors

        side.instance_colors = self.id_colors
        cap.instance_colors  = self.id_colors_caps
        img = canvas.render(region=(x_fb, y_fb, 1,1), size=(1,1), alpha=False)
        side.instance_colors, cap.instance_colors = old_side, old_cap

        pix = img[0,0]
        if pix.dtype != np.uint8:
            pix = np.round(pix*255).astype(int)
        cid = int(pix[0]) | (int(pix[1])<<8) | (int(pix[2])<<16)
        idx = cid - 1 if cid > 0 else None
        if idx is None or idx >= len(self.sec_idx):
            return None

        sec = self.sec_names[self.sec_idx[idx]]
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
    def __init__(self, swc_path):
        super().__init__()
        self.setWindowTitle("Morphology + Voltage")
        self.statusBar().showMessage("Loading morphology…")

        # VisPy 3D scene
        self.canvas3d = scene.SceneCanvas(keys='interactive',
                                          bgcolor='white', show=False)
        self.view = self.canvas3d.central_widget.add_view()
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

        w = QtWidgets.QWidget()
        hb = QtWidgets.QHBoxLayout(w)
        hb.addWidget(self.canvas3d.native)
        hb.addWidget(self.plot2d)
        self.setCentralWidget(w)
        self.resize(1200,600)

        self.data_parent, self.data_child = Pipe(duplex=True)
        self.cmd_parent,  self.cmd_child  = Pipe(duplex=True)

        self.worker = Process(
            target=neuron_process,
            args=(self.data_child, self.cmd_child, swc_path)
        )
        QtCore.QTimer.singleShot(0, self._start_worker)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(1000 // 30)

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
                    pos, ori, rad, ht, col, sec_names, sec_idx, xlocs = msg
                    self.mgr.set_morphology_arrays(
                        pos, ori, rad, ht, col,
                        sec_names, sec_idx, xlocs
                    )
                    elapsed = time.perf_counter() - self._load_t0
                    self.statusBar().showMessage(f"Loaded in {elapsed:.2f}s")
                    QtCore.QTimer.singleShot(3000, self.statusBar().clearMessage)

                    self.geom_ready = True
                    self.canvas3d.events.mouse_press .connect(self._on_mouse_press)
                    self.canvas3d.events.mouse_release.connect(self._on_mouse_release)
                    self.select(sec_names[0], 0.5)
                    continue

                t, arr = msg
                self.mgr.update_colors(arr, lambda a: np.clip((a+80)/130,0,1))
                self.canvas3d.update()

                if self.selected:
                    sec, xloc = self.selected
                    candidates = [
                        (i, abs(self.mgr.xlocs[i] - xloc))
                        for i in range(len(self.mgr.xlocs))
                        if self.mgr.sec_names[self.mgr.sec_idx[i]] == sec
                    ]
                    if candidates:
                        idx = min(candidates, key=lambda x: x[1])[0]
                        self.trace_t.append(t)
                        self.trace_v.append(arr[idx])
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
        dx = ev.pos[0]-self._mouse_start[0]
        dy = ev.pos[1]-self._mouse_start[1]
        del self._mouse_start
        if dx*dx+dy*dy>self.DRAG_THRESHOLD**2:
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


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    swc = os.path.join("res","Animal_2_Basal_2.CNG.swc")
    app = QtWidgets.QApplication(sys.argv)
    w   = MorphologyViewer(swc)
    w.show()
    vispy_app.run()
