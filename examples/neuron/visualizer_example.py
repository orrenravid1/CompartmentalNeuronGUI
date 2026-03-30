"""
Live NEURON visualizer — minimal example of a single-cell live session loaded from an SWC file.

Patterns shown:
  - load_swc_neuron() to import a morphology file as NEURON sections with 3-D coordinates
  - NeuronSession subclass with build_sections() and setup_model()
  - Multiple current clamps for repeated pulse stimulation
  - build_neuron_app() + run_app() as the two-line launch pattern

Requires: NEURON, res/Animal_2_Basal_2.CNG.swc
Run: python examples/neuron/visualizer_example.py
"""

import os

from neuron import h

from compneurovis import NeuronSession, build_neuron_app, run_app
from compneurovis.neuronutils.swc_utils import load_swc_neuron


class ComplexCellSession(NeuronSession):
    def __init__(self):
        super().__init__(title="Complex cell viewer")

    def build_sections(self):
        # load_swc_neuron imports via NEURON's Import3d and returns sections with pt3d coordinates set.
        curr_path = os.path.dirname(os.path.abspath(__file__))
        swc_path = os.path.join(curr_path, "..", "..", "res", "Animal_2_Basal_2.CNG.swc")
        return load_swc_neuron(swc_path)

    def setup_model(self, sections):
        for sec in sections:
            sec.insert("hh")
            # Higher nseg for non-soma sections gives better spatial resolution for propagation.
            if "soma" not in sec.name():
                sec.nseg = 10

        soma = next(sec for sec in sections if "soma" in sec.name().lower())
        # Five brief current pulses spaced across the simulation to drive repeated action potentials.
        self.iclamps = []
        for delay, dur, amp in [(2, 5, 1), (20, 5, 1), (40, 5, 1), (60, 5, 1), (80, 5, 1)]:
            clamp = h.IClamp(soma(0.5))
            clamp.delay = delay
            clamp.dur = dur
            clamp.amp = amp
            self.iclamps.append(clamp)
        return {"iclamps": self.iclamps}


run_app(build_neuron_app(ComplexCellSession()))
