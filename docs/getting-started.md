---
title: Getting Started
summary: Installation and first-run guide for scientists and contributors using CompNeuroVis.
---

# Getting Started

The fastest way to understand CompNeuroVis is to run one example, then
continue with the matching tutorial or concept docs. If you only need one
recommendation, start with **Static Surface First Look**.

## Install

Base install:

```bash
pip install -e .
```

If you want local docs authoring or PR-readiness checks:

```bash
pip install -e ".[contrib]"
```

If you want matplotlib-backed colormaps such as `mpl:viridis`:

```bash
pip install -e ".[matplotlib]"
```

Optional simulator backends:

```bash
pip install -e ".[neuron]"
```

```bash
pip install -e ".[jaxley]"
```

If you want contributor tooling plus a simulator backend in one environment:

```bash
pip install -e ".[matplotlib,neuron]"
pip install -e ".[contrib,jaxley]"
```

The current frontend is a local PyQt6/VisPy desktop app, so examples should be run in a normal GUI session.

## Choose Your First Example

### Static Surface First Look

Run:

```bash
python examples/surface_plot/static_surface_visualizer.py
```

Use this if you want the lowest-friction first run. It renders a 3-D sinc surface with appearance controls and requires no simulator backend.
Then continue with [Build a static surface](tutorials/build-a-static-surface.md).

### Surface Plus Linked Cross-Section

Run:

```bash
python examples/surface_plot/surface_cross_section_visualizer.py
```

Use this if you want a 3-D surface linked to a line plot through shared controls.
It uses a reusable `GridSliceOperatorSpec` so the slice logic is not owned by the surface view itself.
Then continue with [Build a static surface](tutorials/build-a-static-surface.md) and the generated [Example Index](reference/example-index.md) for adjacent variants.

### Custom Backend With Your Own Solver

Run:

```bash
python examples/custom/fitzhugh_nagumo_backend.py
```

Use this if you want the closest runnable reference for "I have my own model and solver."
This example subclasses `BufferedSession` directly, owns a small RK4 integrator, builds
`Scene` / `Field` / `LinePlotViewSpec` / `PanelSpec` manually, and handles controls and actions
without using the NEURON or Jaxley helpers.
When you need timestamped perf logs for debugging, prefer
`AppSpec(diagnostics=DiagnosticsSpec(perf_log_enabled=True))` over shell-only
environment variables.
Then continue with [Session/update model](concepts/session-update-model.md) and the generated [Example Index](reference/example-index.md).

### Custom LIF Point Neuron

Run:

```bash
python examples/custom/lif_backend.py
```

Use this if you want a smaller event-driven custom backend reference for a
spiking neuron. This example subclasses `BufferedSession` directly, owns the
threshold/reset logic and pulse-injection action, and streams explicit
voltage/current/event `Field` histories into linked `LinePlotViewSpec`
panels without using the NEURON or Jaxley helpers.
Then continue with [Session/update model](concepts/session-update-model.md)
and the generated [Example Index](reference/example-index.md).

### Live HH Point-Model With Clamp Control

Run:

```bash
python examples/neuron/hh_point_model_controls.py
```

Requires:

```bash
pip install -e ".[neuron]"
```

Use this if you want the smallest live NEURON example with no morphology panel.
It keeps one `IClamp` active for the full run, so the amplitude slider changes
the injected current live while the voltage trace and `m`/`h`/`n` state
variables keep streaming.
Then continue with [Build a NEURON session](tutorials/build-a-neuron-session.md)
and the generated [Example Index](reference/example-index.md).

### Live Signaling Cascade With Bundled Mechanisms

Run:

```bash
python examples/neuron/signaling_cascade_vis.py
```

Requires:

```bash
pip install -e ".[neuron]"
```

Before first run, compile from inside `examples/neuron/signaling_cascade_mod/`
with your NEURON mechanism compiler. In practice that means running
`nrnivmodl.bat .` on Windows or `nrnivmodl .` on Unix/macOS from that
directory. The example then calls `neuron.load_mechanisms(...)` on the same
folder and loads the resulting `nrnmech.dll` on Windows or the
platform-specific `libnrnmech.*` build directory on Unix/macOS.

Use this if you want a NEURON-backed biochemical/signaling toy model with no
morphology panel and multiple linked traces driven by custom point processes.
Then continue with the generated [Example Index](reference/example-index.md).

### Complex Cell Morphology Viewer

Run:

```bash
python examples/neuron/complex_cell_example.py
```

Requires:

```bash
pip install -e ".[neuron]"
```

Use this if you want a live SWC-backed morphology view with traces.
Then continue with [Build a NEURON session](tutorials/build-a-neuron-session.md).

### Live Jaxley Multicell Example

Run:

```bash
python examples/jaxley/multicell_example.py
```

Requires:

```bash
pip install -e ".[jaxley]"
```

Use this if you want a live procedurally built multicell simulation with synaptic connectivity.
Then continue with [Build a Jaxley session](tutorials/build-a-jaxley-session.md).

### Replay a Precomputed Animation

Run:

```bash
python examples/surface_plot/animated_surface_replay.py
```

Use this if your data already exists as frames and you want to play them back through the same frontend.
Then continue with [Build a replay app](tutorials/build-a-replay-app.md).

## After Your First Run

- Read [Build a static surface](tutorials/build-a-static-surface.md) if you started with static or grid-based field examples.
- Read [Build a NEURON session](tutorials/build-a-neuron-session.md) if you started with the NEURON workflow.
- Read [Build a Jaxley session](tutorials/build-a-jaxley-session.md) if you started with the Jaxley workflow.
- Read [Build a replay app](tutorials/build-a-replay-app.md) if you started from precomputed frames.
- Browse the generated [Example Index](reference/example-index.md) if you want more runnable entrypoints.
- Return to [Docs home](index.md) if you want the broader docs map.

## Local Docs Commands

Serve the docs site locally:

```bash
python -m mkdocs serve
```

Build the docs site in strict mode:

```bash
python -m mkdocs build --strict
```

Published docs deploy through GitHub Pages from the repo's Actions workflow.
After GitHub Pages is configured to use `GitHub Actions`, pushes to `main`
publish the strict MkDocs build automatically at
`https://orrenravid1.github.io/CompNeuroVis/`.

## Contributor PR Flow

1. Run `python scripts/pr_readiness.py check` while iterating locally.
2. Commit your implementation changes normally.
3. As the last commit before you push to `main` or open a PR, run `python scripts/pr_readiness.py seal --commit`.

`seal --commit` reruns the readiness checks, writes a commit-keyed receipt under `.compneurovis/pr-readiness/`, and adds one final attestation commit automatically.
