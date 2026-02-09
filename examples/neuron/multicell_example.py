import time
from neuron import h

from compneurovis.morphology_vis import run_visualizer
from compneurovis.neuron_simulation import NeuronSimulation
from compneurovis.neuronutils.layout import generate_layout


def make_straight_cell(name):
    """
    Cell 1: straight chain.
    soma ─── dend
    """
    soma = h.Section(name=f"{name}_soma")
    soma.L = 20
    soma.diam = 20
    soma.nseg = 1

    dend = h.Section(name=f"{name}_dend")
    dend.L = 200
    dend.diam = 3
    dend.nseg = 10
    dend.connect(soma(1))

    axon = h.Section(name=f"{name}_axon")
    axon.L = 300
    axon.diam = 1.5
    axon.nseg = 10
    axon.connect(soma(0))

    return [soma, dend, axon]


def make_y_cell(name):
    """
    Cell 2: Y-shaped.
            ┌─ dend_a
    soma ───┤
            └─ dend_b
    """
    soma = h.Section(name=f"{name}_soma")
    soma.L = 25
    soma.diam = 15
    soma.nseg = 1

    dend_a = h.Section(name=f"{name}_dend_a")
    dend_a.L = 150
    dend_a.diam = 2.5
    dend_a.nseg = 10
    dend_a.connect(soma(1))

    dend_b = h.Section(name=f"{name}_dend_b")
    dend_b.L = 180
    dend_b.diam = 2.5
    dend_b.nseg = 10
    dend_b.connect(soma(1))

    axon = h.Section(name=f"{name}_axon")
    axon.L = 250
    axon.diam = 1.0
    axon.nseg = 10
    axon.connect(soma(0))

    return [soma, dend_a, dend_b, axon]


def make_branching_cell(name):
    """
    Cell 3: deeper branching.
                    ┌─ branch_a
    soma ─── dend ──┤
                    │        ┌─ twig_a
                    └─ branch_b ──┤
                                  └─ twig_b
    """
    soma = h.Section(name=f"{name}_soma")
    soma.L = 30
    soma.diam = 18
    soma.nseg = 1

    dend = h.Section(name=f"{name}_dend")
    dend.L = 120
    dend.diam = 4
    dend.nseg = 10
    dend.connect(soma(1))

    branch_a = h.Section(name=f"{name}_branch_a")
    branch_a.L = 100
    branch_a.diam = 2.5
    branch_a.nseg = 10
    branch_a.connect(dend(1))

    branch_b = h.Section(name=f"{name}_branch_b")
    branch_b.L = 80
    branch_b.diam = 2.5
    branch_b.nseg = 10
    branch_b.connect(dend(1))

    twig_a = h.Section(name=f"{name}_twig_a")
    twig_a.L = 60
    twig_a.diam = 1.5
    twig_a.nseg = 5
    twig_a.connect(branch_b(1))

    twig_b = h.Section(name=f"{name}_twig_b")
    twig_b.L = 50
    twig_b.diam = 1.5
    twig_b.nseg = 5
    twig_b.connect(branch_b(1))

    axon = h.Section(name=f"{name}_axon")
    axon.L = 200
    axon.diam = 1.2
    axon.nseg = 10
    axon.connect(soma(0))

    return [soma, dend, branch_a, branch_b, twig_a, twig_b, axon]


class MultiCellSimulation(NeuronSimulation):

    def __init__(self):
        super().__init__()
        self.secs = None

    @property
    def sections(self):
        return self.secs

    def setup(self):
        t0 = time.perf_counter()

        # --- Build 3 cells ---
        self.cell1_secs = make_straight_cell("cell1")
        self.cell2_secs = make_y_cell("cell2")
        self.cell3_secs = make_branching_cell("cell3")

        self.secs = self.cell1_secs + self.cell2_secs + self.cell3_secs

        # --- Biophysics: HH on all sections ---
        for sec in self.secs:
            sec.insert("hh")

        # --- Synaptic connections via NetCon + ExpSyn ---
        # Cell 1 axon -> Cell 2 soma
        syn1 = h.ExpSyn(self.cell2_secs[0](0.5))
        syn1.tau = 2.0
        syn1.e = 0.0
        nc1 = h.NetCon(
            self.cell1_secs[2](0.9)._ref_v,  # cell1 axon near tip
            syn1,
            sec=self.cell1_secs[2],
        )
        nc1.weight[0] = 0.05
        nc1.delay = 1.0

        # Cell 2 axon -> Cell 3 soma
        syn2 = h.ExpSyn(self.cell3_secs[0](0.5))
        syn2.tau = 2.0
        syn2.e = 0.0
        nc2 = h.NetCon(
            self.cell2_secs[3](0.9)._ref_v,  # cell2 axon near tip
            syn2,
            sec=self.cell2_secs[3],
        )
        nc2.weight[0] = 0.05
        nc2.delay = 1.0

        # Cell 3 axon -> Cell 1 dend (recurrent)
        syn3 = h.ExpSyn(self.cell1_secs[1](0.5))
        syn3.tau = 2.0
        syn3.e = 0.0
        nc3 = h.NetCon(
            self.cell3_secs[6](0.9)._ref_v,  # cell3 axon near tip
            syn3,
            sec=self.cell3_secs[6],
        )
        nc3.weight[0] = 0.03
        nc3.delay = 1.0

        # Store to prevent garbage collection
        self.synapses = [syn1, syn2, syn3]
        self.netcons = [nc1, nc2, nc3]

        # --- Current injection into cell 1 soma to drive the network ---
        self.iclamps = []
        for delay, dur, amp in [(2, 5, 0.5), (20, 5, 0.5), (40, 5, 0.5),
                                (60, 5, 0.5), (80, 5, 0.5)]:
            icl = h.IClamp(self.cell1_secs[0](0.5))
            icl.delay = delay
            icl.dur = dur
            icl.amp = amp
            self.iclamps.append(icl)

        elapsed = time.perf_counter() - t0
        print(f"Cells built in {elapsed:.2f}s")

        # --- Layout: position cells so they don't overlap ---
        # cell_connections places cell2's soma at the end of cell1's axon,
        # and cell3's soma at the end of cell2's axon.
        generate_layout(
            self.secs,
            cell_connections=[
                (self.cell2_secs[0], self.cell1_secs[2], 1.0),  # cell2 soma at cell1 axon tip
                (self.cell3_secs[0], self.cell2_secs[3], 1.0),  # cell3 soma at cell2 axon tip
            ],
        )
        # Uses the default layout which ignores connectivity
        ##h.define_shape()


run_visualizer(MultiCellSimulation())
