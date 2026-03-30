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

from compneurovis import JaxleySession, build_jaxley_app, run_app
from compneurovis.jaxleyutils import translate_cells_xyzr


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
        super().__init__(title="Multi-cell Jaxley viewer", dt=0.1, display_dt=0.5)

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

    def setup_model(self, network, cells):
        for cell in cells:
            cell.insert(HH())

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
        network.select(edges="all").set("IonotropicSynapse_gS", 5e-4)

        pulses = [(2.0, 5.0, 0.5), (20.0, 5.0, 0.5), (40.0, 5.0, 0.5), (60.0, 5.0, 0.5), (80.0, 5.0, 0.5)]
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
        network.cell(0).branch(0).loc(0.5).stimulate(current)
        return {"stimulus": current}


run_app(build_jaxley_app(MultiCellSession()))
