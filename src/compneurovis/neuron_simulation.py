#!/usr/bin/env python3
import time
import numpy as np
from abc import abstractmethod
from neuron import h

from compneurovis.simulation import Simulation


class NeuronSimulation(Simulation):
    dt: float
    v_init: float
    _morph_recorders: dict[str, tuple[any, any]]
    _sim_recorders: dict[str, tuple[any, any]]

    def __init__(self, dt=0.1, v_init=-65):
        super().__init__()
        # Will map morphology_var→(PtrVector,Vector)
        self._morph_recorders   = {}
        # Will map sim_var→(PtrVector,Vector)
        self._sim_recorders = {}
        self.dt = dt
        self.v_init = v_init
    
    @property
    @abstractmethod
    def sections(self):
        pass

    def build_morphology_meta(self):
        """
        Pure-NumPy, vectorized builder for per-segment NEURON metadata.
        Returns a dict with:
        - positions (M,3)
        - orientations (M,3,3)
        - radii (M,)
        - lengths (M,)
        - colors (M,4)
        - sec_names (list of str)
        - sec_idx (M,)
        - xloc (M,)
        """
        t0 = time.perf_counter()
        # gather per‐section data
        sec_names = []
        P0, P1, D0, D1 = [], [], [], []
        CUM, TOT, S = [], [], []

        for si, sec in enumerate(self.sections):
            n3d = int(sec.n3d())
            if n3d < 2:
                continue
            sec_names.append(sec.name())

            # extract coords & diameters
            pts   = np.stack([[sec.x3d(i), sec.y3d(i), sec.z3d(i)] for i in range(n3d)], axis=0).astype(np.float32)
            diams = np.array([sec.diam3d(i) for i in range(n3d)], dtype=np.float32)

            # segment vectors & lengths
            diffs = pts[1:] - pts[:-1]                    # (n3d-1,3)
            dlen  = np.linalg.norm(diffs, axis=1)         # (n3d-1,)
            cum   = np.concatenate(([0.0], np.cumsum(dlen)))[:-1]  # (n3d-1,)
            total = cum[-1] + dlen[-1] if dlen.sum()>0 else 1.0

            # record
            P0.append(pts[:-1])
            P1.append(pts[1:])
            D0.append(diams[:-1])
            D1.append(diams[1:])
            CUM.append(cum)
            TOT.append(np.full_like(dlen, total, dtype=np.float32))
            S .append(np.full_like(dlen, si,    dtype=np.int32))

        # stack into flat arrays
        P0   = np.vstack(P0)         # (M,3)
        P1   = np.vstack(P1)
        D0   = np.concatenate(D0)    # (M,)
        D1   = np.concatenate(D1)
        CUM  = np.concatenate(CUM)
        TOT  = np.concatenate(TOT)
        S    = np.concatenate(S)
        M    = P0.shape[0]

        # compute midpoints, lengths, radii, normalized xloc
        mid   = 0.5 * (P0 + P1)                    # (M,3)
        L     = np.linalg.norm(P1 - P0, axis=1)    # (M,)
        xloc  = (CUM + 0.5 * L) / TOT              # (M,)
        rad   = 0.5 * (D0 + D1)                    # (M,)
        col   = np.tile(np.array([0.7,0.7,0.7,1.0], dtype=np.float32), (M,1))
        
        # orientations via bulk Rodrigues
        diffs = P1 - P0                           # (M,3)
        L     = np.linalg.norm(diffs, axis=1)     # (M,)
        dn    = np.zeros_like(diffs)              # (M,3)
        # handling zero or near zero lengths
        nz    = L > 1e-8
        dn[nz] = diffs[nz] / L[nz,None]
        cos_t = dn[:,2]                             # dot with z
        ang   = np.arccos(np.clip(cos_t, -1.0, 1.0))# (M,)
        ax    = np.cross(np.repeat([[0,0,1]], M, 0), dn)  # (M,3)
        ax_n  = np.linalg.norm(ax, axis=1, keepdims=True)
        ax_u  = np.divide(ax, ax_n, where=(ax_n>1e-6))
        ux, uy, uz = ax_u.T

        # build skew K and K²
        K    = np.zeros((M,3,3), dtype=np.float32)
        K[:,0,1] = -uz; K[:,0,2] =  uy
        K[:,1,0] =  uz; K[:,1,2] = -ux
        K[:,2,0] = -uy; K[:,2,1] =  ux
        K2   = K @ K  # (M,3,3)

        sin_t = np.sin(ang)[:,None,None]
        one_c = (1.0 - cos_t)[:,None,None]
        I     = np.eye(3, dtype=np.float32)[None,:,:]  # broadcastable

        R = I + sin_t * K + one_c * K2  # (M,3,3)

        elapsed = time.perf_counter() - t0
        print(f"Meta file generated in {elapsed:.2f}s")

        return {
            'positions':    mid.astype(np.float32),
            'orientations': R.astype(np.float32),
            'radii':        rad.astype(np.float32),
            'lengths':      L.astype(np.float32),
            'colors':       col,
            'sec_names':    sec_names,
            'sec_idx':      S,
            'xloc':         xloc.astype(np.float32)
        }
    
    def initialize(self):
        h.dt = self.dt
        h.finitialize(self.v_init)

    # TODO: More generic recording
    def record(self):
        self.record_simulation_vars('t')
        self.record_morphology_vars('v')
    
    def get_data(self, *args, **kwargs):
        data = {}
        # Just return them all in one dictionary
        for (varname, (pvs, vs)) in self._morph_recorders.items():
            if args and varname not in args:
                continue
            else:
                pvs.gather(vs)
                arr = vs.as_numpy()
                data[varname] = arr
        for (varname, (pv, v)) in self._sim_recorders.items():
            if args and varname not in args:
                continue
            else:
                pv.gather(v)
                arr = v.as_numpy()
                data[varname] = arr[0]
        return data

    def record_morphology_vars(self, *args, **kwargs):
        idxs  = self.morphology_meta["sec_idx"]
        xlocs = self.morphology_meta["xloc"]
        for varname in args:
            if varname not in self._morph_recorders:
                pvs = h.PtrVector(self.morphology_count)
                vs = h.Vector(self.morphology_count)
                for i,(si,x) in enumerate(zip(idxs, xlocs)):
                    sec = self.sections[si]
                    # e.g. getattr(sec(x), "_ref_v") or "_ref_cai"
                    ref = getattr(sec(x), f"_ref_{varname}")
                    pvs.pset(i, ref)

                self._morph_recorders[varname] = (pvs, vs)

    def record_simulation_vars(self, *args, **kwargs):
        for varname in args:
            if varname not in self._sim_recorders:
                pv = h.PtrVector(1)
                v = h.Vector(1)
                ref = getattr(h, f"_ref_{varname}")
                pv.pset(0, ref)

                self._sim_recorders[varname] = (pv, v)
        

                
            

