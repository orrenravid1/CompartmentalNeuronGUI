import os

from neuron import h

from compneurovis import NeuronSession, build_neuron_app, run_app
from compneurovis.neuronutils.swc_utils import load_swc_neuron


class ComplexCellSession(NeuronSession):
    def __init__(self):
        super().__init__(title="Complex cell viewer")

    def build_sections(self):
        curr_path = os.path.dirname(os.path.abspath(__file__))
        swc_path = os.path.join(curr_path, "..", "..", "res", "Animal_2_Basal_2.CNG.swc")
        return load_swc_neuron(swc_path)

    def setup_model(self, sections):
        for sec in sections:
            sec.insert("hh")
            if "soma" not in sec.name():
                sec.nseg = 10

        soma = next(sec for sec in sections if "soma" in sec.name().lower())
        self.iclamps = []
        for delay, dur, amp in [(2, 5, 1), (20, 5, 1), (40, 5, 1), (60, 5, 1), (80, 5, 1)]:
            clamp = h.IClamp(soma(0.5))
            clamp.delay = delay
            clamp.dur = dur
            clamp.amp = amp
            self.iclamps.append(clamp)
        return {"iclamps": self.iclamps}


run_app(build_neuron_app(ComplexCellSession()))
