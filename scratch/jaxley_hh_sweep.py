"""scratch/jaxley_hh_sweep.py — HH amplitude sweep with matplotlib output.

Caches each run to scratch/hh_sweep_cache/<amp>.npy so runs survive restarts.
Compiles once, sweeps i_amp, saves sweep.png next to this script.
Run: python scratch/jaxley_hh_sweep.py
"""
import os
from jax import config
config.update("jax_enable_x64", True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "hh_sweep_cache")
OUT_PNG = os.path.join(SCRIPT_DIR, "sweep.png")
os.makedirs(CACHE_DIR, exist_ok=True)

DT = 0.025
N_STEPS = 1000
AMPS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]


def cache_path(amp: float) -> str:
    return os.path.join(CACHE_DIR, f"amp_{amp:.3f}.npy")


def run_amp(amp: float, init_fn, step_fn, params, all_params, externals_ref, external_inds, rec_indices) -> np.ndarray:
    import jaxley as jx
    t_max = N_STEPS * DT
    i_ext = jx.step_current(i_delay=0.0, i_dur=t_max, i_amp=amp, delta_t=DT, t_max=t_max)
    externals = {}
    for k, a in externals_ref.items():
        externals[k] = i_ext.reshape(a.shape) if k == "i" else a

    state, _ = init_fn(params, delta_t=DT)
    voltages = np.empty(N_STEPS, dtype=np.float32)
    for i in range(N_STEPS):
        ext = {}
        for k, a in externals.items():
            ext[k] = a[..., i] if a.ndim > 1 else (a[i] if i < a.shape[0] else a[-1])
        state = step_fn(state, all_params, ext, external_inds, delta_t=DT)
        voltages[i] = float(np.asarray(state["v"])[rec_indices[0]])
        if (i + 1) % 500 == 0:
            print(f" {(i+1)*DT:.0f}ms", end="", flush=True)
    return voltages


def make_plot(amps, t_axis):
    fig, axes = plt.subplots(len(amps), 1, figsize=(10, 2.2 * len(amps)), sharex=True)
    if len(amps) == 1:
        axes = [axes]
    for ax, amp in zip(axes, amps):
        cp = cache_path(amp)
        voltages = np.load(cp)
        n_spikes = int(np.sum(np.diff((voltages > 0.0).astype(int)) == 1))
        ax.plot(t_axis, voltages, lw=0.8, color="steelblue")
        ax.set_ylabel("v (mV)", fontsize=8)
        ax.set_title(
            f"i_amp={amp:.2f} nA  |  spikes={n_spikes}  |  "
            f"v_min={voltages.min():.1f}  v_max={voltages.max():.1f}",
            fontsize=8,
        )
        ax.axhline(0, color="k", lw=0.4, ls="--")
        ax.set_ylim(-90, 80)
    axes[-1].set_xlabel("t (ms)")
    fig.suptitle("Jaxley HH — amplitude sweep (single compartment, v_init=-70 mV)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Saved {OUT_PNG}")


t_axis = np.arange(1, N_STEPS + 1) * DT
missing = [amp for amp in AMPS if not os.path.exists(cache_path(amp))]

if not missing:
    print("All runs cached. Regenerating plot...")
    make_plot(AMPS, t_axis)
else:
    import jax
    import jaxley as jx
    from jaxley.channels import HH
    from jaxley.integrate import build_init_and_step_fn

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Cached: {[a for a in AMPS if os.path.exists(cache_path(a))]}")
    print(f"Need to run: {missing}\n")

    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=1)
    xyzr = [np.array([[0.0, 0.0, 0.0, 10.0], [20.0, 0.0, 0.0, 10.0]], dtype=np.float32)]
    cell = jx.Cell([branch], parents=[-1], xyzr=xyzr)
    cell.insert(HH())
    network = jx.Network([cell])

    t_max = N_STEPS * DT
    network.stimulate(jx.step_current(i_delay=0.0, i_dur=t_max, i_amp=AMPS[0], delta_t=DT, t_max=t_max))
    network.record("v", verbose=False)
    network.set("v", -70.0)
    network.init_states()
    network.to_jax()
    params = network.get_parameters()
    externals_ref = {k: np.asarray(v) for k, v in network.externals.items()}
    external_inds = {k: np.asarray(v) for k, v in network.external_inds.items()}
    rec_indices = np.asarray(network.recordings.rec_index.to_numpy(), dtype=np.int32)

    print("Compiling step fn...")
    init_fn, step_fn = build_init_and_step_fn(network)
    state0, all_params = init_fn(params, delta_t=DT)
    _ext0 = {k: (a[..., 0] if a.ndim > 1 else a[0]) for k, a in externals_ref.items()}
    _ = step_fn(state0, all_params, _ext0, external_inds, delta_t=DT)
    print("JIT ready.\n")

    for amp in AMPS:
        cp = cache_path(amp)
        if os.path.exists(cp):
            print(f"i_amp={amp:.2f} nA — loaded from cache")
            continue
        print(f"Running i_amp={amp:.2f} nA ...", end="", flush=True)
        voltages = run_amp(amp, init_fn, step_fn, params, all_params, externals_ref, external_inds, rec_indices)
        np.save(cp, voltages)
        n_spikes = int(np.sum(np.diff((voltages > 0.0).astype(int)) == 1))
        print(f"  done. v_min={voltages.min():.1f} v_max={voltages.max():.1f} spikes={n_spikes}")

    make_plot(AMPS, t_axis)
