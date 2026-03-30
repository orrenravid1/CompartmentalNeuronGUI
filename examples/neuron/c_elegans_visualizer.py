"""
C. elegans morphology visualizer — loads a directory of multi-tree SWC files as a single session.

Patterns shown:
  - load_swc_multi() for SWC files containing multiple disconnected trees (one per neurite type)
  - Building sections from an entire directory of per-cell SWC files
  - Stimulating all somas independently with randomised pulse timing

Requires: NEURON, res/celegans_cells_swc/ directory of SWC files
Run: python examples/neuron/c_elegans_visualizer.py
"""

import os
import random

from neuron import h

from compneurovis import NeuronSession, build_neuron_app, run_app
from compneurovis.neuronutils.swc_utils import load_swc_multi


class CElegansSession(NeuronSession):
    def __init__(self):
        super().__init__(title="C. elegans morphology viewer")

    def build_sections(self):
        curr_path = os.path.dirname(os.path.abspath(__file__))
        swc_path = os.path.join(curr_path, "..", "..", "res", "celegans_cells_swc")
        sections = []
        for swc_file in os.listdir(swc_path):
            print(f"Loading cell {swc_file}")
            cell_name = swc_file.split(".")[0]
            # load_swc_multi handles SWC files with multiple disconnected trees (e.g. separate
            # axon and dendrite trees). Returns a dict of tree_id → section list; we flatten all.
            trees = load_swc_multi(os.path.join(swc_path, swc_file), cell_name)
            for section_list in trees.values():
                sections.extend(section_list)
        return sections

    def setup_model(self, sections):
        for sec in sections:
            sec.insert("hh")
            if "soma" not in sec.name():
                sec.nseg = 10

        somas = [sec for sec in sections if "soma" in sec.name().lower()]
        self.iclamps = []
        for soma in somas:
            # Randomised jitter on pulse timing prevents all cells firing in perfect synchrony.
            for delay, dur, amp in [
                (2 + random.random() * 5, 5, 0.2),
                (20 + random.random() * 5, 5, 0.2),
                (40 + random.random() * 5, 5, 0.2),
                (60 + random.random() * 5, 5, 0.2),
                (80 + random.random() * 5, 5, 0.2),
            ]:
                clamp = h.IClamp(soma(0.5))
                clamp.delay = delay
                clamp.dur = dur
                clamp.amp = amp
                self.iclamps.append(clamp)
        return {"iclamps": self.iclamps}


run_app(build_neuron_app(CElegansSession))
