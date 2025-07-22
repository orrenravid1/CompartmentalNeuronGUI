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


def build_morphology_meta_jaxley(cells):
    import numpy as np
    from jaxley.io.graph import to_swc_graph

    sec_names, branch_id = [], {}
    bid = 0
    # 1) assign each (cell,branch) a global branch ID & name
    for ci, cell in enumerate(cells):
        df_names = to_swc_graph(cell)[['global_branch_index','name']].drop_duplicates()
        for _, row in df_names.iterrows():
            branch_id[(ci, int(row.global_branch_index))] = bid
            sec_names.append(row.name)
            bid += 1

    # 2) collect segment endpoints, diameters, and branch IDs
    P0s, P1s, D0s, D1s = [], [], [], []
    CUMs, TOTs, Ss     = [], [], []

    for ci, cell in enumerate(cells):
        df = to_swc_graph(cell)
        for b, sub in df.groupby('global_branch_index'):
            sub = sub.sort_values('local_comp_index')
            pts   = sub[['x','y','z']].to_numpy(np.float32)
            diams = sub['r'].to_numpy(np.float32)
            if len(pts) < 2: 
                continue

            diffs = pts[1:] - pts[:-1]
            dlen  = np.linalg.norm(diffs, axis=1)
            cum   = np.concatenate(([0.0], np.cumsum(dlen)))[:-1]
            total= cum[-1] + dlen[-1] if dlen.sum()>0 else 1.0

            P0s.append(pts[:-1]); P1s.append(pts[1:])
            D0s.append(diams[:-1]); D1s.append(diams[1:])
            CUMs.append(cum)
            TOTs.append(np.full_like(cum, total, dtype=np.float32))
            Ss.append(np.full_like(cum, branch_id[(ci,b)], dtype=np.int32))

    # 3) stack & filter zero‐length
    P0  = np.vstack(P0s);   P1 = np.vstack(P1s)
    D0  = np.concatenate(D0s); D1 = np.concatenate(D1s)
    CUM = np.concatenate(CUMs); TOT = np.concatenate(TOTs)
    S   = np.concatenate(Ss)
    L   = np.linalg.norm(P1 - P0, axis=1)
    mask= L > 1e-6

    mid   = 0.5*(P0[mask] + P1[mask])
    rad   = 0.5*(D0[mask] + D1[mask])
    col   = np.tile([0.7,0.7,0.7,1.0], (len(mid),1)).astype(np.float32)
    xloc  = (CUM[mask] + 0.5*L[mask]) / TOT[mask]

    # 4) Rodrigues rotations
    dn    = (P1[mask] - P0[mask]) / L[mask,None]
    cos_t = dn[:,2]
    ang   = np.arccos(np.clip(cos_t, -1, 1))
    ax    = np.cross(np.repeat([[0,0,1]], len(dn), 0), dn)
    ax_n  = np.linalg.norm(ax, axis=1, keepdims=True)
    ax_u  = ax / np.where(ax_n>1e-6, ax_n, 1.0)
    ux, uy, uz = ax_u.T

    K    = np.zeros((len(dn),3,3), dtype=np.float32)
    K[:,0,1] = -uz; K[:,0,2] =  uy
    K[:,1,0] =  uz; K[:,1,2] = -ux
    K[:,2,0] = -uy; K[:,2,1] =  ux
    K2   = K @ K

    sin_t = np.sin(ang)[:,None,None]
    one_c = (1 - cos_t)[:,None,None]
    I     = np.eye(3, dtype=np.float32)[None,:,:]
    R     = I + sin_t*K + one_c*K2

    return {
        'positions':    mid,
        'orientations': R,
        'radii':        rad,
        'lengths':      L[mask],
        'colors':       col,
        'sec_names':    sec_names,
        'sec_idx':      S[mask],
        'xloc':         xloc
    }


# ——— Monkey‐patch Jaxley’s radius_from_xyzr to stub zero‐radius compartments ———
import jaxley.utils.cell_utils as _cu

# Save original
_orig_radius_from_xyzr = _cu.radius_from_xyzr

# Override: if average radius ≤ 0, use constant min_radius for all segments
def _patched_radius_from_xyzr(xyzr, min_radius):
    avg = np.mean(xyzr[:, 3])
    if avg <= 0.0:
        return np.full(xyzr.shape[0], min_radius, dtype=np.float32)
    return _orig_radius_from_xyzr(xyzr, min_radius)

# Apply patch
_cu.radius_from_xyzr = _patched_radius_from_xyzr


def jaxley_process(data_pipe, cmd_pipe):
    import os, numpy as np
    from jax import config
    import jaxley as jx
    from jaxley.channels import Leak, HH
    from jaxley.integrate import build_init_and_step_fn, add_stimuli
    from jaxley.io.graph import to_swc_graph

    # — Configure JAX for CPU (or GPU/TPU) —
    config.update("jax_enable_x64", True)
    config.update("jax_platform_name", "cpu")

    # — Load & sanitize SWCs into Jaxley cells (min_radius=1.0) —
    swc_dir   = "res/celegans_cells_swc"
    swc_files = sorted(os.listdir(swc_dir))
    cells = []
    for fn in swc_files:
        path = os.path.join(swc_dir, fn)
        cell = jx.read_swc(path, ncomp=10, min_radius=1.0)
        cells.append(cell)

    # — Insert Leak only in dendritic branches, HH elsewhere —
    for cell in cells:
        df = to_swc_graph(cell)
        dend_br = set(df[df['apical'] | df['basal']]['global_branch_index'].unique())
        for b in range(cell.num_branches):
            mech = Leak() if b in dend_br else HH()
            cell.branch(b).insert(mech)

    # — Build & send morphology meta (use your existing build_morphology_meta_jaxley) —
    meta = build_morphology_meta_jaxley(cells)
    data_pipe.send(meta)

    # — Create five‐pulse step‐current and set up recordings —
    dt     = 0.1
    pulses = [(2,5,0.2), (20,5,0.2), (40,5,0.2), (60,5,0.2), (80,5,0.2)]
    t_max  = pulses[-1][0] + pulses[-1][1]
    current = sum(
        jx.step_current(i_delay=d, i_dur=dur, i_amp=amp,
                        delta_t=dt,   t_max=t_max)
        for d, dur, amp in pulses
    )

    net = jx.Network(cells)
    net.delete_stimuli(); net.delete_recordings()
    for c in cells:
        c.stimulate(current)
        for b in range(c.num_branches):
            c.branch(b).loc(0.5).record("v")

    # — Initialize & step (step_fn ≡ h.fadvance) —
    params    = net.get_parameters()
    init_fn, step_fn = build_init_and_step_fn(net)
    state, all_params = init_fn(params, delta_t=dt)

    externals, external_inds = add_stimuli(
        net.externals.copy(), net.external_inds.copy(),
        (None, current, None)
    )
    rec_inds   = net.recordings.rec_index.to_numpy()
    rec_states = net.recordings.state.to_numpy()

    nsteps = int(t_max // dt) + 1
    for step in range(nsteps):
        t = float(step * dt)
        ext = {k: externals[k][step] for k in externals}
        state = step_fn(state, all_params, ext,
                        external_inds, delta_t=dt)
        vs = np.array([state[s][i] for s, i in zip(rec_states, rec_inds)], dtype=np.float32)
        data_pipe.send((t, vs))

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
            target=jaxley_process,
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
