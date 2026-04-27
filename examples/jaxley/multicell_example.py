"""
Multi-cell Jaxley visualizer - three procedurally-built cells connected by synapses.

Patterns shown:
  - Programmatic Jaxley cell construction with explicit branch xyzr coordinates
  - Feedforward ring connectivity via IonotropicSynapse
  - Repeated step-current stimulation on the first cell
  - translate_cells_xyzr() to space cells apart before building the network

Requires: jax, jaxley
Run: python examples/jaxley/multicell_example.py
"""

import numpy as np

import jaxley as jx
from jaxley.channels import HH
from jaxley.connect import connect
from jaxley.synapses import IonotropicSynapse

from compneurovis import ControlPresentationSpec, ControlSpec, JaxleySession, ScalarValueSpec, build_jaxley_app, run_app
from compneurovis.backends.jaxley.utils import translate_cells_xyzr


def float_control(
    control_id: str,
    label: str,
    default: float,
    min_value: float,
    max_value: float,
    steps: int,
    *,
    scale: str = "linear",
) -> ControlSpec:
    return ControlSpec(
        id=control_id,
        label=label,
        value_spec=ScalarValueSpec(default=default, min=min_value, max=max_value, value_type="float"),
        presentation=ControlPresentationSpec(kind="slider", steps=steps, scale=scale),
        send_to_session=True,
    )


def make_straight_cell(name: str):
    comp = jx.Compartment()
    branches = [
        jx.Branch(comp, ncomp=4),
        jx.Branch(comp, ncomp=8),
        jx.Branch(comp, ncomp=10),
    ]
    xyzr = [
        np.array([[0.0, 0.0, 0.0, 10.0], [18.0, 0.0, 0.0, 10.0]], dtype=np.float32),
        np.array([[18.0, 0.0, 0.0, 2.0], [120.0, 28.0, 0.0, 2.0]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.0, 1.2], [-150.0, 0.0, 0.0, 1.2]], dtype=np.float32),
    ]
    cell = jx.Cell(branches, parents=[-1, 0, 0], xyzr=xyzr)
    cell.meta_name = name
    return cell


def make_y_cell(name: str):
    comp = jx.Compartment()
    branches = [
        jx.Branch(comp, ncomp=4),
        jx.Branch(comp, ncomp=7),
        jx.Branch(comp, ncomp=7),
        jx.Branch(comp, ncomp=10),
    ]
    xyzr = [
        np.array([[0.0, 0.0, 0.0, 8.0], [16.0, 0.0, 0.0, 8.0]], dtype=np.float32),
        np.array([[16.0, 0.0, 0.0, 2.2], [88.0, 44.0, 0.0, 2.2]], dtype=np.float32),
        np.array([[16.0, 0.0, 0.0, 2.2], [88.0, -44.0, 0.0, 2.2]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.0, 1.0], [-140.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    ]
    cell = jx.Cell(branches, parents=[-1, 0, 0, 0], xyzr=xyzr)
    cell.meta_name = name
    return cell


def make_branching_cell(name: str):
    comp = jx.Compartment()
    branches = [
        jx.Branch(comp, ncomp=4),
        jx.Branch(comp, ncomp=6),
        jx.Branch(comp, ncomp=5),
        jx.Branch(comp, ncomp=5),
        jx.Branch(comp, ncomp=4),
        jx.Branch(comp, ncomp=4),
        jx.Branch(comp, ncomp=9),
    ]
    xyzr = [
        np.array([[0.0, 0.0, 0.0, 9.0], [18.0, 0.0, 0.0, 9.0]], dtype=np.float32),
        np.array([[18.0, 0.0, 0.0, 3.0], [70.0, 18.0, 0.0, 3.0]], dtype=np.float32),
        np.array([[70.0, 18.0, 0.0, 2.0], [120.0, 52.0, 0.0, 2.0]], dtype=np.float32),
        np.array([[70.0, 18.0, 0.0, 2.0], [120.0, -10.0, 0.0, 2.0]], dtype=np.float32),
        np.array([[120.0, -10.0, 0.0, 1.2], [150.0, 18.0, 0.0, 1.2]], dtype=np.float32),
        np.array([[120.0, -10.0, 0.0, 1.2], [150.0, -34.0, 0.0, 1.2]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.0, 1.0], [-120.0, -8.0, 0.0, 1.0]], dtype=np.float32),
    ]
    cell = jx.Cell(branches, parents=[-1, 0, 1, 1, 3, 3, 0], xyzr=xyzr)
    cell.meta_name = name
    return cell


class MultiCellSession(JaxleySession):
    def __init__(self):
        super().__init__(title="Multi-cell Jaxley viewer", dt=0.1)
        self.stim_amp = 0.5
        self.stim_dur = 5.0
        self.syn_gs = 5e-4
        self.syn_e_syn = 0.0
        self.syn_k_minus = 0.5
        self.syn_v_th = -35.0
        self.syn_delta = 10.0
        self.hh_gna = 0.12
        self.hh_gk = 0.036
        self.hh_gleak = 3e-4

    def morphology_color_limits(self):
        return (-80.0, 50.0)

    def control_specs(self):
        return {
            "display_dt": float_control("display_dt", "Visual update interval (ms sim/update)", self.display_dt, self.dt, 5.0, 98),
            "stim_amp": float_control("stim_amp", "Stimulus amplitude (nA)", self.stim_amp, 0.0, 5.0, 200),
            "stim_dur": float_control("stim_dur", "Stimulus duration (ms)", self.stim_dur, 1.0, 20.0, 190),
            "syn_gs": float_control("syn_gs", "Synapse gS", self.syn_gs, 1e-5, 1e-1, 200, scale="log"),
            "syn_e_syn": float_control("syn_e_syn", "Synapse reversal (mV)", self.syn_e_syn, -80.0, 60.0, 280),
            "syn_k_minus": float_control("syn_k_minus", "Synapse k_minus", self.syn_k_minus, 1e-3, 1.0, 200, scale="log"),
            "syn_v_th": float_control("syn_v_th", "Synapse pre-V threshold (mV)", self.syn_v_th, -70.0, 10.0, 160),
            "syn_delta": float_control("syn_delta", "Synapse pre-V slope (mV)", self.syn_delta, 1.0, 20.0, 190),
            "hh_gna": float_control("hh_gna", "HH gNa (S/cm^2)", self.hh_gna, 0.01, 0.3, 200),
            "hh_gk": float_control("hh_gk", "HH gK (S/cm^2)", self.hh_gk, 0.005, 0.12, 200),
            "hh_gleak": float_control("hh_gleak", "HH gLeak (S/cm^2)", self.hh_gleak, 1e-5, 3e-3, 200, scale="log"),
        }

    def build_cells(self):
        cells = [
            make_straight_cell("cell1"),
            make_y_cell("cell2"),
            make_branching_cell("cell3"),
        ]
        translate_cells_xyzr(
            cells,
            offsets=[
                (0.0, 0.0, 0.0),
                (220.0, 120.0, 0.0),
                (440.0, -40.0, 0.0),
            ],
        )
        return cells

    def _pulse_schedule(self):
        return [(2.0, self.stim_dur, self.stim_amp), (20.0, self.stim_dur, self.stim_amp), (40.0, self.stim_dur, self.stim_amp), (60.0, self.stim_dur, self.stim_amp), (80.0, self.stim_dur, self.stim_amp)]

    def _apply_hh_parameters(self, network):
        network.set("HH_gNa", self.hh_gna)
        network.set("HH_gK", self.hh_gk)
        network.set("HH_gLeak", self.hh_gleak)

    def _apply_synapse_parameters(self, network):
        edge_view = network.select(edges="all")
        edge_view.set("IonotropicSynapse_gS", self.syn_gs)
        edge_view.set("IonotropicSynapse_e_syn", self.syn_e_syn)
        edge_view.set("IonotropicSynapse_k_minus", self.syn_k_minus)
        edge_view.set("IonotropicSynapse_v_th", self.syn_v_th)
        edge_view.set("IonotropicSynapse_delta", self.syn_delta)

    def _initialize_synapse_state(self, network):
        network.select(edges="all").set("IonotropicSynapse_s", 0.0)

    def _rebuild_stimulus(self, network=None):
        target = network if network is not None else self.network
        if target is None:
            return
        target.delete_stimuli()
        pulses = self._pulse_schedule()
        t_max = pulses[-1][0] + pulses[-1][1] + 10.0
        current = sum(
            jx.step_current(
                i_delay=delay,
                i_dur=dur,
                i_amp=amp,
                delta_t=self.dt,
                t_max=t_max,
            )
            for delay, dur, amp in pulses
        )
        target.cell(0).branch(0).loc(0.5).stimulate(current)
        if network is None:
            self.refresh_runtime_externals()

    def setup_model(self, network, cells):
        network.insert(HH())
        self._apply_hh_parameters(network)

        # Feedforward ring: cell1 axon -> cell2 soma -> cell3 soma -> cell1 dendrite.
        connect(
            network.cell(0).branch(2).comp(9),
            network.cell(1).branch(0).comp(1),
            IonotropicSynapse(),
        )
        connect(
            network.cell(1).branch(3).comp(9),
            network.cell(2).branch(0).comp(1),
            IonotropicSynapse(),
        )
        connect(
            network.cell(2).branch(6).comp(8),
            network.cell(0).branch(1).comp(4),
            IonotropicSynapse(),
        )
        self._apply_synapse_parameters(network)
        self._initialize_synapse_state(network)
        self._rebuild_stimulus(network)
        return {"stimulus": network.externals.get("i")}

    def apply_control(self, control_id: str, value) -> bool:
        if control_id == "display_dt":
            self.display_dt = max(self.dt, float(value))
            return True
        if control_id == "stim_amp":
            self.stim_amp = float(value)
            self._rebuild_stimulus()
            return True
        if control_id == "stim_dur":
            self.stim_dur = float(value)
            self._rebuild_stimulus()
            return True
        if control_id == "syn_gs":
            self.syn_gs = float(value)
            if self.network is not None:
                self._apply_synapse_parameters(self.network)
                self.refresh_runtime_parameters()
            return True
        if control_id == "syn_e_syn":
            self.syn_e_syn = float(value)
            if self.network is not None:
                self._apply_synapse_parameters(self.network)
                self.refresh_runtime_parameters()
            return True
        if control_id == "syn_k_minus":
            self.syn_k_minus = float(value)
            if self.network is not None:
                self._apply_synapse_parameters(self.network)
                self.refresh_runtime_parameters()
            return True
        if control_id == "syn_v_th":
            self.syn_v_th = float(value)
            if self.network is not None:
                self._apply_synapse_parameters(self.network)
                self.refresh_runtime_parameters()
            return True
        if control_id == "syn_delta":
            self.syn_delta = float(value)
            if self.network is not None:
                self._apply_synapse_parameters(self.network)
                self.refresh_runtime_parameters()
            return True
        if control_id == "hh_gna":
            self.hh_gna = float(value)
            if self.network is not None:
                self._apply_hh_parameters(self.network)
                self.refresh_runtime_parameters()
            return True
        if control_id == "hh_gk":
            self.hh_gk = float(value)
            if self.network is not None:
                self._apply_hh_parameters(self.network)
                self.refresh_runtime_parameters()
            return True
        if control_id == "hh_gleak":
            self.hh_gleak = float(value)
            if self.network is not None:
                self._apply_hh_parameters(self.network)
                self.refresh_runtime_parameters()
            return True
        return super().apply_control(control_id, value)

if __name__ == "__main__":
    run_app(build_jaxley_app(MultiCellSession))
