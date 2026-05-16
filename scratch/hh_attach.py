"""scratch/hh_attach.py — single-compartment HH model via attach() API.

attach() wraps an existing NEURON model without subclassing NeuronBackend.
Morphology view + voltage trace + IClamp control.

Requires: NEURON
Run: python scratch/hh_attach.py
"""
from neuron import h
from compneurovis.backends.neuron.attach import attach

soma = h.Section(name="soma")
soma.L = 12.6157
soma.diam = 12.6157
soma.nseg = 1
soma.insert("hh")

stim = h.IClamp(soma(0.5))
stim.delay = 0.0
stim.dur = 1e9
stim.amp = 0.1

h.dt = 0.025
h.celsius = 6.3
h.finitialize(-65.0)

app = attach(sections=[soma])
app.control(
    "clamp_amp",
    label="IClamp amplitude (nA)",
    get=lambda: float(stim.amp),
    set=lambda v: setattr(stim, "amp", float(v)),
    min=-0.2,
    max=0.5,
)
app.show(title="HH attach")
