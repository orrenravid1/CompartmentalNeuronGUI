import time
import os
from neuron import h

from compneurovis.morphology_vis import run_visualizer
from compneurovis.neuron_simulation import NeuronSimulation
from compneurovis.neuronutils.swc_utils import load_swc_neuron


class ComplexCellNeuronSimulation(NeuronSimulation):

    def __init__(self):
        super().__init__()
        self.secs = None
    
    @property
    def sections(self):
        return self.secs

    def setup(self):
        t0 = time.perf_counter()
        curr_path = os.path.dirname(os.path.abspath(__file__))
        swc_path = os.path.join(curr_path,"..","..","res","Animal_2_Basal_2.CNG.swc")
        self.secs = load_swc_neuron(swc_path)

        elapsed = time.perf_counter() - t0
        print(f"SWC Loaded in {elapsed:.2f}s")

        for sec in self.secs:
            sec.insert("hh" if "dendrite" not in sec.name() else "pas")
            if "soma" not in sec.name():
                sec.nseg = 10

        soma = next(sec for sec in self.secs if 'soma' in sec.name().lower())
        self.iclamps = []
        for d,du,a in [(2,5,1),(20,5,1),(40,5,1),(60,5,5),(80,5,5)]:
            icl = h.IClamp(soma(0.5))
            icl.delay, icl.dur, icl.amp = d, du, a
            self.iclamps.append(icl)

    def step(self):
        h.fadvance()
  
run_visualizer(ComplexCellNeuronSimulation())