#!/usr/bin/env python3
import sys
import os
import numpy as np
import multiprocessing as mp
from multiprocessing.managers import BaseManager

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import Qt
from vispy import use
use(app='pyqt6', gl='gl+')
from vispy import scene, app as vispy_app
from vispy.scene.cameras import TurntableCamera
from scipy.spatial.transform import Rotation
import pyqtgraph as pg

from neuron import h
from src.vispyutils.cappedcylindercollection import CappedCylinderCollection

# ─────────────────────────────────────────────────────────────────────────────
def build_morphology_meta(secs):
    t_neurite = np.array([0.7,0.7,0.7,1.0], dtype=np.float32)
    meta = []
    for sec in secs:
        name = sec.name()
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
            z   = np.array([0,0,1], dtype=np.float32)
            dn  = (p1 - p0) / L
            ax  = np.cross(z, dn)
            if np.linalg.norm(ax) < 1e-6:
                R = np.eye(3, dtype=np.float32)
            else:
                ax /= np.linalg.norm(ax)
                ang = np.arccos(np.clip(np.dot(z, dn), -1,1))
                R   = Rotation.from_rotvec(ax*ang).as_matrix().astype(np.float32)
            midlen = cum[i] + 0.5*d[i]
            xloc   = float(midlen / total)
            meta.append({
                'position':    mid,
                'radius':      rad,
                'length':      L,
                'orientation': R,
                'color':       t_neurite,
                'sec_name':    name,
                'xloc':        xloc
            })
    return meta

def load_swc_model(swc_path):
    h.load_file("stdlib.hoc")
    h.load_file("import3d.hoc")
    r = h.Import3d_SWC_read(); r.input(swc_path)
    gui = h.Import3d_GUI(r, False); gui.instantiate(None)
    return list(h.allsec())

# ─────────────────────────────────────────────────────────────────────────────
class NeuronSim:
    def __init__(self, swc_path):
        secs = load_swc_model(swc_path)
        for sec in secs:
            mech = "pas" if "dendrite" in sec.name() else "hh"
            sec.insert(mech)
        self.meta = build_morphology_meta(secs)

        h.dt = 0.1
        self.tvec = h.Vector(); self.tvec.record(h._ref_t)

        self.vs = []
        name2sec = {sec.name(): sec for sec in secs}
        for m in self.meta:
            sec = name2sec[m['sec_name']]
            vec = h.Vector(); vec.record(sec(m['xloc'])._ref_v)
            self.vs.append(vec)

        self.iclamps = []
        soma = next(sec for sec in secs if 'soma' in sec.name().lower())
        for d,du,a in [(2,10,1),(20,10,1),(40,10,1),(60,20,1)]:
            icl = h.IClamp(soma(0.5))
            icl.delay, icl.dur, icl.amp = d, du, a
            self.iclamps.append(icl)

        h.finitialize(-65.0)

    def step(self, n=1):
        for _ in range(n):
            h.fadvance()

    def get_time(self):
        return float(self.tvec[-1])

    def get_voltages(self):
        return np.array([v[-1] for v in self.vs], dtype=np.float32)

    def get_meta(self):
        return self.meta

# register with manager
class NeuronManager(BaseManager): pass
NeuronManager.register('NeuronSim', NeuronSim)

# ─────────────────────────────────────────────────────────────────────────────
class SimWorker(QtCore.QObject):
    data_ready = QtCore.pyqtSignal(float, np.ndarray)
    error      = QtCore.pyqtSignal(Exception)

    @QtCore.pyqtSlot(int)
    def step_and_fetch(self, n):
        try:
            sim.step(n)
            t  = sim.get_time()
            vs = sim.get_voltages()
            self.data_ready.emit(t, vs)
        except Exception as e:
            self.error.emit(e)

# ─────────────────────────────────────────────────────────────────────────────
class MorphologyManager:
    def __init__(self, view):
        self.view = view
        self.meta = []
        self.collection = None
        self.id_colors = None
        self.id_colors_caps   = None
        self.orig_side = None
        self.orig_cap  = None

    @staticmethod
    def make_id_color(i):
        cid = i+1
        return np.array([ (cid&0xFF)/255.0,
                          ((cid>>8)&0xFF)/255.0,
                          ((cid>>16)&0xFF)/255.0,
                          1.0 ], dtype=np.float32)

    def set_morphology(self, meta):
        self.meta = meta
        N = len(meta)
        pos  = np.stack([m['position']    for m in meta])
        rad  = np.array([m['radius']      for m in meta], dtype=np.float32)
        ht   = np.array([m['length']      for m in meta], dtype=np.float32)
        ori  = np.stack([m['orientation'] for m in meta])
        cols = np.stack([m['color']       for m in meta])

        self.collection = CappedCylinderCollection(
            positions=pos, radii=rad, heights=ht,
            orientations=ori, colors=cols,
            cylinder_segments=32, disk_slices=32,
            parent=self.view.scene
        )
        self.collection._side_mesh.shading = None
        self.collection._cap_mesh.shading  = None

        self.id_colors = np.stack([self.make_id_color(i) for i in range(N)])
        self.id_colors_caps   = np.vstack([self.id_colors, self.id_colors])
        self.orig_side = self.collection._side_mesh.instance_colors.copy()
        self.orig_cap  = self.collection._cap_mesh.instance_colors.copy()

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


    def update_colors(self, vs):
        norm = np.clip((vs+80)/130,0,1)
        cols = np.zeros((len(vs),4),dtype=np.float32)
        cols[:,0]=norm; cols[:,1]=0.2; cols[:,2]=1-norm; cols[:,3]=1
        self.collection.set_colors(cols)

# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, swc):
        super().__init__()
        self.setWindowTitle("Integrated RPC + GUI")

        # start manager, create sim proxy
        self.manager = NeuronManager(address=('localhost',0), authkey=b'neu')
        self.manager.start()
        global sim
        sim = self.manager.NeuronSim(swc)

        # VisPy setup
        self.canvas3d = scene.SceneCanvas(keys='interactive', bgcolor='white', show=False)
        self.view     = self.canvas3d.central_widget.add_view()
        self.view.camera = TurntableCamera(fov=60, distance=200,
                                           elevation=30, azimuth=30)

        # geometry
        self.mmgr = MorphologyManager(self.view)
        self.mmgr.set_morphology(sim.get_meta())

        # click selection
        self.selected = None
        self.DRAG_THRESHOLD = 5
        self.canvas3d.events.mouse_press .connect(self._press)
        self.canvas3d.events.mouse_release.connect(self._release)

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
        self.resize(1000,600)

        # worker thread
        self.thread = QtCore.QThread(self)
        self.worker = SimWorker()
        self.worker.moveToThread(self.thread)
        self.worker.data_ready.connect(self._update)
        self.worker.error     .connect(lambda e: print("Worker error:",e))
        self.thread.start()

        # timer at 60 Hz
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(lambda: self.worker.step_and_fetch(1))
        self.timer.start(1000//60)

    def _press(self, ev):
        self._p0 = ev.pos

    def _release(self, ev):
        if not hasattr(self,'_p0'): return
        dx = ev.pos[0]-self._p0[0]; dy = ev.pos[1]-self._p0[1]
        del self._p0
        if dx*dx+dy*dy>self.DRAG_THRESHOLD**2: return
        x,y = ev.pos; w,h=self.canvas3d.size; ps=self.canvas3d.pixel_scale
        xf, yf = int(x*ps), int((h-y-1)*ps)
        picked = self.mmgr.pick(xf,yf,self.canvas3d)
        if picked:
            self.selected = picked
            self.trace_t.clear(); self.trace_v.clear()
            self.trace.clear()
            self.plot2d.setTitle(f"Voltage for {picked[0]}@{picked[1]:.3f}")

    @QtCore.pyqtSlot(float, np.ndarray)
    def _update(self, t, vs):
        # 3D
        self.mmgr.update_colors(vs)
        self.canvas3d.update()
        # 2D
        if self.selected:
            sec, sel_x = self.selected
            cand = [(i,abs(m['xloc']-sel_x)) for i,m in enumerate(self.mmgr.meta)
                    if m['sec_name']==sec]
            if cand:
                idx = min(cand, key=lambda x:x[1])[0]
                self.trace_t.append(t); self.trace_v.append(vs[idx])
                if len(self.trace_t)>5000:
                    self.trace_t.pop(0); self.trace_v.pop(0)
                self.trace.setData(self.trace_t, self.trace_v, clear=False)

    def closeEvent(self, ev):
        self.timer.stop()
        self.thread.quit(); self.thread.wait()
        sim = None
        self.manager.shutdown()
        super().closeEvent(ev)

if __name__=="__main__":
    mp.set_start_method('spawn', force=True)
    swc = os.path.join("res","Animal_2_Basal_2.CNG.swc")
    app = QtWidgets.QApplication(sys.argv)
    w   = MainWindow(swc)
    w.show()
    vispy_app.run()
