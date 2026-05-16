"""scratch/jaxley_hh_validate.py — validate Jaxley HH dynamics, no UI.

Prints voltage every 10 steps so you can watch in real time.
Run: python scratch/jaxley_hh_validate.py
"""
import numpy as np
import jax
import jaxley as jx
from jaxley.channels import HH
from jaxley.integrate import build_init_and_step_fn

DT = 0.025
N_STEPS = 1000  # 100 ms total
I_AMP = 0.15   # nA — between rheobase and dep-block

print(f"JAX backend: {jax.default_backend()}")
print(f"i_amp = {I_AMP} nA  |  dt = {DT} ms  |  T = {N_STEPS*DT:.0f} ms\n")

comp = jx.Compartment()
branch = jx.Branch(comp, ncomp=1)
xyzr = [np.array([[0.0, 0.0, 0.0, 10.0], [20.0, 0.0, 0.0, 10.0]], dtype=np.float32)]
cell = jx.Cell([branch], parents=[-1], xyzr=xyzr)
cell.insert(HH())
network = jx.Network([cell])

i_ext = jx.step_current(i_delay=0.0, i_dur=N_STEPS * DT, i_amp=I_AMP, delta_t=DT, t_max=N_STEPS * DT)
network.stimulate(i_ext)
network.record("v", verbose=False)
network.set("v", -70.0)
network.init_states()
network.to_jax()
params = network.get_parameters()

init_fn, step_fn = build_init_and_step_fn(network)
state, all_params = init_fn(params, delta_t=DT)

externals = {k: np.asarray(v) for k, v in network.externals.items()}
external_inds = {k: np.asarray(v) for k, v in network.external_inds.items()}
rec_indices = np.asarray(network.recordings.rec_index.to_numpy(), dtype=np.int32)

print(f"Initial v = {float(np.asarray(state['v'])[rec_indices[0]]):.2f} mV")
print(f"\n{'step':>6}  {'t (ms)':>8}  {'v (mV)':>10}")

v_min, v_max = 1e9, -1e9
n_spikes = 0
prev_above = False

for i in range(N_STEPS):
    ext = {k: v[i] if v.ndim == 1 and i < v.shape[0] else (v[..., i] if v.ndim > 1 and i < v.shape[-1] else np.zeros_like(v[0] if v.ndim == 1 else v[..., 0])) for k, v in externals.items()}
    state = step_fn(state, all_params, ext, external_inds, delta_t=DT)
    v = float(np.asarray(state['v'])[rec_indices[0]])
    v_min = min(v_min, v)
    v_max = max(v_max, v)
    above = v > 0.0
    if above and not prev_above:
        n_spikes += 1
    prev_above = above
    if i % 10 == 0:
        print(f"{i:>6}  {(i+1)*DT:>8.3f}  {v:>10.3f}")

print(f"\nv range: min={v_min:.2f} mV  max={v_max:.2f} mV  spikes={n_spikes}")
