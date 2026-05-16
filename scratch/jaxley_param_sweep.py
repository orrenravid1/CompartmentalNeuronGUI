"""scratch/jaxley_param_sweep.py — sweep compartment radius and cm at fixed i_amp.

Caches runs to scratch/param_sweep_cache_biophysical_geometry/.
Saves param_sweep.png next to script.
Run: python scratch/jaxley_param_sweep.py
"""
import os
from jax import config
config.update("jax_enable_x64", True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(SCRIPT_DIR, "param_sweep_cache_biophysical_geometry")
OUT_PNG    = os.path.join(SCRIPT_DIR, "param_sweep.png")
os.makedirs(CACHE_DIR, exist_ok=True)

DT      = 0.025
N_STEPS = 2000   # 50 ms
I_AMP   = 0.10   # nA, fixed

# Sweep: (radius_um, length_um, cm_uF_per_cm2, label)
SWEEP = [
    (1.0,  10.0, 1.0,  "r=1µm  L=10µm  cm=1.0"),
    (2.0,  10.0, 1.0,  "r=2µm  L=10µm  cm=1.0"),
    (5.0,  20.0, 1.0,  "r=5µm  L=20µm  cm=1.0"),
    (10.0, 20.0, 1.0,  "r=10µm L=20µm  cm=1.0"),
    (10.0, 20.0, 0.5,  "r=10µm L=20µm  cm=0.5"),
    (10.0, 20.0, 2.0,  "r=10µm L=20µm  cm=2.0"),
    (20.0, 40.0, 1.0,  "r=20µm L=40µm  cm=1.0"),
]


def cache_path(radius, length, cm):
    return os.path.join(CACHE_DIR, f"r{radius:.1f}_L{length:.1f}_cm{cm:.2f}.npy")


def build_and_run(radius, length, cm):
    import jaxley as jx
    from jaxley.channels import HH
    from jaxley.integrate import build_init_and_step_fn

    t_max = N_STEPS * DT
    comp  = jx.Compartment()
    comp.set("length", length)
    comp.set("radius", radius)
    branch = jx.Branch(comp, ncomp=1)
    xyzr = [np.array([[0.0, 0.0, 0.0, radius],
                       [length, 0.0, 0.0, radius]], dtype=np.float64)]
    cell = jx.Cell([branch], parents=[-1], xyzr=xyzr)
    cell.insert(HH())
    network = jx.Network([cell])

    if cm != 1.0:
        network.set("capacitance", cm)

    i_ext = jx.step_current(i_delay=0.0, i_dur=t_max, i_amp=I_AMP,
                             delta_t=DT, t_max=t_max)
    network.stimulate(i_ext)
    network.record("v", verbose=False)
    network.set("v", -70.0)
    network.init_states()
    network.to_jax()
    params = network.get_parameters()
    external_inds = {k: np.asarray(v) for k, v in network.external_inds.items()}
    rec_indices   = np.asarray(network.recordings.rec_index.to_numpy(), dtype=np.int32)

    init_fn, step_fn = build_init_and_step_fn(network)
    state, all_params = init_fn(params, delta_t=DT)
    externals = {k: np.asarray(v) for k, v in network.externals.items()}

    voltages = np.empty(N_STEPS, dtype=np.float64)
    for i in range(N_STEPS):
        ext = {}
        for k, a in externals.items():
            ext[k] = a[..., i] if a.ndim > 1 else (a[i] if i < a.shape[0] else a[-1])
        state = step_fn(state, all_params, ext, external_inds, delta_t=DT)
        voltages[i] = float(np.asarray(state["v"])[rec_indices[0]])
        if (i + 1) % 400 == 0:
            print(f"  {(i+1)*DT:.0f}ms", end="", flush=True)
    return voltages.astype(np.float32)


def make_plot():
    t_axis = np.arange(1, N_STEPS + 1) * DT
    fig, axes = plt.subplots(len(SWEEP), 1, figsize=(12, 2.2 * len(SWEEP)), sharex=True)
    for ax, (radius, length, cm, label) in zip(axes, SWEEP):
        voltages = np.load(cache_path(radius, length, cm))
        n_spikes = int(np.sum(np.diff((voltages > 0.0).astype(int)) == 1))
        area_um2 = 2 * np.pi * radius * length
        density  = (I_AMP * 1e-9 / (area_um2 * 1e-8)) * 1e6  # µA/cm²
        ax.plot(t_axis, voltages, lw=0.7, color="steelblue")
        ax.set_ylabel("v (mV)", fontsize=8)
        ax.set_title(
            f"{label}  |  area={area_um2:.0f}µm²  I_density={density:.1f}µA/cm²  "
            f"spikes={n_spikes}  v∈[{voltages.min():.0f},{voltages.max():.0f}]mV",
            fontsize=8,
        )
        ax.axhline(0, color="k", lw=0.4, ls="--")
        ax.set_ylim(-90, 80)
    axes[-1].set_xlabel("t (ms)")
    fig.suptitle(f"Jaxley HH — param sweep  i_amp={I_AMP} nA  T={N_STEPS*DT:.0f} ms", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nSaved {OUT_PNG}")


missing = [(r, l, cm, lbl) for r, l, cm, lbl in SWEEP if not os.path.exists(cache_path(r, l, cm))]

if not missing:
    print("All cached. Regenerating plot...")
    make_plot()
else:
    import jax
    print(f"JAX backend: {jax.default_backend()}  x64={jax.config.x64_enabled}")
    print(f"Need to run {len(missing)}/{len(SWEEP)} configs\n")

    for radius, length, cm, label in SWEEP:
        cp = cache_path(radius, length, cm)
        if os.path.exists(cp):
            print(f"CACHED  {label}")
            continue
        print(f"Running {label} ...", end="", flush=True)
        v = build_and_run(radius, length, cm)
        np.save(cp, v)
        n_spikes = int(np.sum(np.diff((v > 0.0).astype(int)) == 1))
        print(f"  v∈[{v.min():.1f},{v.max():.1f}] spikes={n_spikes}")

    make_plot()
