import time
import os
from neuron import h

from compneurovis.morphology_vis import run_visualizer
from compneurovis.neuron_simulation import NeuronSimulation
from compneurovis.neuronutils.swc_utils import load_swc_multi


class CElegansNeuronSimulation(NeuronSimulation):

    def __init__(self):
        super().__init__()
        self.secs = None
    
    @property
    def sections(self):
        return self.secs

    def setup(self):
        t0 = time.perf_counter()
        curr_path = os.path.dirname(os.path.abspath(__file__))
        swc_path = os.path.join(curr_path,"..","..","res","celegans_cells_swc")
        swc_files = [f for f in os.listdir(swc_path)]
        self.secs = []
        for i,swcf in enumerate(swc_files):
            print(f"Loading cell {swcf}")
            cell_name = swcf.split('.')[0]
            trees = load_swc_multi(os.path.join(swc_path, swcf), cell_name)
            seclists = trees.values()
            for seclist in seclists:
                for sec in seclist:
                    self.secs.append(sec)

        elapsed = time.perf_counter() - t0
        print(f"SWCs Loaded in {elapsed:.2f}s")

        for sec in self.secs:
            sec.insert("hh" if "dendrite" not in sec.name() else "pas")
            if "soma" not in sec.name():
                sec.nseg = 10

        somas = [sec for sec in self.secs if 'soma' in sec.name().lower()]
        # WARNING: Need to store iclamps outside of this method i.e. via self otherwise they will
        # be garbage collected
        self.iclamps = []
        for soma in somas:
            for d,du,a in [(2,5,0.2),(20,5,0.2),(40,5,0.2),(60,5,0.2),(80,5,0.2)]:
                icl = h.IClamp(soma(0.5))
                icl.delay, icl.dur, icl.amp = d, du, a
                self.iclamps.append(icl)
                

run_visualizer(CElegansNeuronSimulation())