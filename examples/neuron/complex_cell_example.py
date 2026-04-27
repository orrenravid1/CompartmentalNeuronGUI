"""
Complex cell visualizer — minimal example of a single-cell live session loaded from an SWC file.

Patterns shown:
  - load_swc_neuron() to import a morphology file as NEURON sections with 3-D coordinates
  - NeuronSession subclass with build_sections() and setup_model()
  - Multiple current clamps for repeated pulse stimulation
  - build_neuron_app() + run_app() as the two-line launch pattern

Requires: NEURON, res/Animal_2_Basal_2.CNG.swc
Run: python examples/neuron/complex_cell_example.py
"""

import os

from neuron import h

from compneurovis import ControlPresentationSpec, ControlSpec, NeuronSession, ScalarValueSpec, build_neuron_app, run_app
from compneurovis.backends.neuron.utils import load_swc_neuron


class ComplexCellSession(NeuronSession):
    def __init__(self):
        super().__init__(title="Complex cell viewer")
        self.stim_amp = 1.0

    def build_sections(self):
        # load_swc_neuron imports via NEURON's Import3d and returns sections with pt3d coordinates set.
        curr_path = os.path.dirname(os.path.abspath(__file__))
        swc_path = os.path.join(curr_path, "..", "..", "res", "Animal_2_Basal_2.CNG.swc")
        return load_swc_neuron(swc_path)

    def control_specs(self):
        return {
            "display_dt": ControlSpec(
                id="display_dt",
                label="Visual update interval (ms sim/update)",
                value_spec=ScalarValueSpec(default=self.display_dt, min=self.dt, max=5.0, value_type="float"),
                presentation=ControlPresentationSpec(kind="slider", steps=98),
                send_to_session=True,
            ),
            "stim_amp": ControlSpec(
                id="stim_amp",
                label="Stimulus amplitude (nA)",
                value_spec=ScalarValueSpec(default=self.stim_amp, min=0.0, max=2.0, value_type="float"),
                presentation=ControlPresentationSpec(kind="slider", steps=100),
                send_to_session=True,
            ),
        }

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
            clamp.amp = self.stim_amp * amp
            self.iclamps.append(clamp)
        return {"iclamps": self.iclamps}

    def apply_control(self, control_id: str, value) -> bool:
        if control_id == "display_dt":
            self.display_dt = max(self.dt, float(value))
            return True
        if control_id == "stim_amp":
            self.stim_amp = float(value)
            for clamp in getattr(self, "iclamps", []):
                clamp.amp = self.stim_amp
            return True
        return super().apply_control(control_id, value)


run_app(build_neuron_app(ComplexCellSession))
