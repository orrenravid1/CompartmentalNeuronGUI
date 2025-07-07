#!/usr/bin/env python3
import sys
import os
import numpy as np
import multiprocessing as mp
from multiprocessing import Pipe, Process

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import Qt
from vispy import use
# ─── VisPy + PySide6 backend ────────────────────────────────────────────────
use(app='pyqt6', gl='gl+')
from vispy import scene, app
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
        color = t_neurite
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
                'color':       color,
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

def neuron_process(data_pipe, cmd_pipe, swc_path):
    """
    Worker process:
    - reload SWC so NEURON has sections
    - builds its own meta
    - steps NEURON in tight loops, sending every h.t advance
    - listens for 'reset' and reinitializes
    """
    from neuron import h

    # reload morphology so NEURON can simulate it
    secs = load_swc_model(swc_path)
    for sec in secs:
        if "dendrite" not in sec.name():
            sec.insert("hh")
        else:
            sec.insert("pas")
    print(f"[worker] Loaded SWC with {len(secs)} sections")
    meta = build_morphology_meta(secs)
    print(f"[worker] Built meta with {len(meta)} segments")

    # set up recording
    name_to_sec = {sec.name(): sec for sec in secs}
    refs = [(name_to_sec[m['sec_name']], m['xloc']) for m in meta]
    N = len(refs)

    h.dt = 0.01
    vec_t = h.Vector(); vec_t.record(h._ref_t)
    vecs_v = []
    for sec, xloc in refs:
        vvec = h.Vector(); vvec.record(sec(xloc)._ref_v)
        vecs_v.append(vvec)

    # persist IClamps
    soma = next(sec for sec in secs if 'soma' in sec.name().lower())
    iclamps = []
    for delay, dur, amp in [(2,10,1),(20,10,1),(40,10,1),(60,20,1)]:
        icl = h.IClamp(soma(0.5))
        icl.delay, icl.dur, icl.amp = delay, dur, amp
        iclamps.append(icl)

    h.finitialize(-65.0)

    try:
        while True:
            h.fadvance()
            # handle reset
            while cmd_pipe.poll():
                cmd = cmd_pipe.recv()
                if cmd == "reset":
                    h.finitialize(-65.0)
            # send one datapoint per advance
            t_last = float(vec_t[-1])
            v_last = np.empty(N, dtype=np.float32)
            for i, vvec in enumerate(vecs_v):
                v_last[i] = vvec[-1]
            data_pipe.send((t_last, v_last))
    finally:
        data_pipe.close()
        cmd_pipe.close()

class MorphologyViewer(QtWidgets.QMainWindow):
    def __init__(self, swc_path):
        super().__init__()
        self.setWindowTitle("Morphology + Voltage (selectable segment)")

        # load geometry in parent
        secs = load_swc_model(swc_path)
        print("Parent loaded SWC sections:", [sec.name() for sec in secs])
        self.meta = build_morphology_meta(secs)
        print(f"Parent built meta with {len(self.meta)} segments")
        N = len(self.meta)

        # VisPy 3D scene
        self.canvas3d = scene.SceneCanvas(keys='interactive',
                                          bgcolor='white', show=False)
        self.view = self.canvas3d.central_widget.add_view()
        self.view.camera = TurntableCamera(fov=60, distance=200,
                                           elevation=30, azimuth=30,
                                           translate_speed=100, up='+z')

        # instanced cylinders
        pos  = np.stack([m['position']    for m in self.meta], axis=0)
        rad  = np.array([m['radius']      for m in self.meta], dtype=np.float32)
        ht   = np.array([m['length']      for m in self.meta], dtype=np.float32)
        ori  = np.stack([m['orientation'] for m in self.meta], axis=0)
        cols = np.stack([m['color']       for m in self.meta], axis=0)
        self.collection = CappedCylinderCollection(
            positions=pos, radii=rad, heights=ht,
            orientations=ori, colors=cols,
            cylinder_segments=32, disk_slices=32,
            parent=self.view.scene
        )
        # disable lighting for picking
        self.collection._side_mesh.shading = None
        self.collection._cap_mesh.shading  = None

        # prepare GPU-picking colors
        def make_id_color(i):
            cid = i + 1
            return np.array([
                ( cid        & 0xFF) / 255.0,
                ((cid >>  8) & 0xFF) / 255.0,
                ((cid >> 16) & 0xFF) / 255.0,
                1.0
            ], dtype=np.float32)
        self.id_colors      = np.stack([make_id_color(i) for i in range(N)], axis=0)
        self.id_colors_caps = np.vstack([self.id_colors, self.id_colors])
        # save original colors
        self.orig_side_colors = self.collection._side_mesh.instance_colors.copy()
        self.orig_cap_colors  = self.collection._cap_mesh.instance_colors.copy()

        # selection state
        self.selected_idx  = 0
        self.click_start   = None
        self.DRAG_THRESHOLD = 5

        # hook mouse events
        self.canvas3d.events.mouse_press .connect(self.on_mouse_press)
        self.canvas3d.events.mouse_release.connect(self.on_mouse_release)

        # PyQtGraph plot for the selected segment
        self.plot2d  = pg.PlotWidget(title="Voltage for segment 0")
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

        # prepare pipes and worker—do NOT start immediately
        self.data_parent, data_child = Pipe()
        self.cmd_parent,  cmd_child  = Pipe()
        self.cmd_pipe = self.cmd_parent
        self.worker_proc = Process(
            target=neuron_process,
            args=(data_child, cmd_child, swc_path),
            daemon=True
        )
        # stash the child‐ends so we can close them after starting
        self._worker_args = (data_child, cmd_child)

        # delay spawning until after the window is shown
        QtCore.QTimer.singleShot(0, self._start_worker)

        # timer to pull sim data
        self.u_timer = QtCore.QTimer(self)
        self.u_timer.timeout.connect(self._poll_worker)
        self.u_timer.start(1000 // 30)

    def _start_worker(self):
        """Spawn the NEURON subprocess after the GUI has painted."""
        data_child, cmd_child = self._worker_args
        self.worker_proc.start()
        data_child.close()
        cmd_child.close()

    def keyPressEvent(self, event):
        # catch the spacebar in PySide6
        if event.key() == Qt.Key.Key_Space:
            self.cmd_pipe.send('reset')
            print("Sent reset command to NEURON process")
        super().keyPressEvent(event)

    def on_mouse_press(self, event):
        self.click_start = event.pos

    def on_mouse_release(self, event):
        if self.click_start is None:
            return
        dx = event.pos[0] - self.click_start[0]
        dy = event.pos[1] - self.click_start[1]
        self.click_start = None
        if dx*dx + dy*dy > self.DRAG_THRESHOLD**2:
            return

        x_log, y_log = event.pos
        w_log, h_log = self.canvas3d.size
        ps = self.canvas3d.pixel_scale
        x_fb = int(x_log * ps)
        y_fb = int((h_log - y_log - 1) * ps)

        # render ID-pass
        self.collection._side_mesh .instance_colors = self.id_colors
        self.collection._cap_mesh  .instance_colors = self.id_colors_caps
        img = self.canvas3d.render(region=(x_fb, y_fb, 1, 1), size=(1, 1), alpha=False)
        # restore
        self.collection._side_mesh .instance_colors = self.orig_side_colors
        self.collection._cap_mesh  .instance_colors = self.orig_cap_colors
        self.canvas3d.update()

        sample = img[0, 0]
        if sample.dtype != np.uint8:
            sample = np.round(sample * 255).astype(int)
        cid = int(sample[0]) | (int(sample[1])<<8) | (int(sample[2])<<16)
        idx = cid - 1 if cid > 0 else None
        if idx is None or not (0 <= idx < len(self.meta)):
            print("No segment hit")
            return

        self.selected_idx = idx
        sec_name = self.meta[idx]['sec_name']
        xloc     = self.meta[idx]['xloc']
        print(f"Selected {sec_name} @ {xloc:.3f} (idx={idx})")
        self.trace_t.clear()
        self.trace_v.clear()
        self.trace.setData(self.trace_t, self.trace_v)
        self.plot2d.setTitle(f"Voltage for {sec_name}@{xloc:.3f}")

    def _poll_worker(self):
        try:
            last = None
            while self.data_parent.poll():
                last = self.data_parent.recv()
            if last is not None:
                t, v = last
                self._apply_update(t, v)
        except (BrokenPipeError, EOFError, OSError):
            self.u_timer.stop()

    def _apply_update(self, t, v_last):
        if self.trace_t and t <= self.trace_t[-1]:
            self.trace_t.clear()
            self.trace_v.clear()

        norm     = np.clip((v_last + 80)/130, 0.0, 1.0)
        new_cols = np.zeros((len(v_last),4), dtype=np.float32)
        new_cols[:,0] = norm
        new_cols[:,1] = 0.2
        new_cols[:,2] = 1.0 - norm
        new_cols[:,3] = 1.0
        self.collection.set_colors(new_cols)

        self.trace_t.append(t)
        self.trace_v.append(v_last[self.selected_idx])
        if len(self.trace_t) > 5000:
            self.trace_t.pop(0)
            self.trace_v.pop(0)
        self.trace.setData(self.trace_t, self.trace_v)

        self.canvas3d.update()

    def closeEvent(self, ev):
        if self.worker_proc.is_alive():
            self.worker_proc.terminate()
            self.worker_proc.join()
        self.u_timer.stop()
        super().closeEvent(ev)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    qtapp = QtWidgets.QApplication(sys.argv)
    viewer = MorphologyViewer(os.path.join("res", "m3s4s4t-vp-sup.CNG.swc"))
    viewer.show()
    app.run()
