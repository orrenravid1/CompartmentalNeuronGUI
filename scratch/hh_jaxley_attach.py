"""scratch/hh_jaxley_attach.py — single-compartment HH via attach() with parameter controls.

Controls: stimulus amplitude, gNa, gK, gLeak.
Run: python scratch/hh_jaxley_attach.py
"""
from jax import config
config.update("jax_enable_x64", True)

import numpy as np
import jaxley as jx
from jaxley.channels import HH

from compneurovis.backends.jaxley.attach import attach

DT = 0.025
STIM_DURATION = 500.0

comp = jx.Compartment()
comp.set("length", 10)
comp.set("radius", 10)
branch = jx.Branch(comp, ncomp=1)
xyzr = [np.array([[0.0, 0.0, 0.0, 10.0], [20.0, 0.0, 0.0, 10.0]], dtype=np.float64)]
cell = jx.Cell([branch], parents=[-1], xyzr=xyzr)
cell.insert(HH())
cell.meta_name = "soma"

_net = [None]
_i_amp = [0.1]


def setup(network, cells):
    _net[0] = network
    network.stimulate(jx.step_current(
        i_delay=0.0, i_dur=STIM_DURATION, i_amp=_i_amp[0],
        delta_t=DT, t_max=STIM_DURATION,
    ))


def set_i_amp(v):
    _i_amp[0] = v
    if _net[0] is not None:
        _net[0].delete_stimuli()
        _net[0].stimulate(jx.step_current(
            i_delay=0.0, i_dur=STIM_DURATION, i_amp=v,
            delta_t=DT, t_max=STIM_DURATION,
        ))


app = attach(cells=[cell], setup=setup, dt=DT, v_init=-70.0)

app.control("i_amp",  label="Stimulus (nA)",    get=lambda: _i_amp[0],                set=set_i_amp,                            min=0.0, max=1.0, refresh_externals=True)
app.control("gNa",    label="gNa (S/cm²)",      get=lambda: 0.12,                     set=lambda v: _net[0].set("HH_gNa", v),   min=0.01, max=0.5,  refresh_params=True)
app.control("gK",     label="gK (S/cm²)",        get=lambda: 0.036,                    set=lambda v: _net[0].set("HH_gK", v),    min=0.005, max=0.2, refresh_params=True)
app.control("gLeak",  label="gLeak (S/cm²)",    get=lambda: 0.0003,                   set=lambda v: _net[0].set("HH_gLeak", v), min=1e-5, max=0.01, refresh_params=True)

app.show(title="HH Jaxley — parameter controls")
