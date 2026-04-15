---
title: Getting Started
summary: Installation and first-run guide for scientists and contributors using CompNeuroVis.
---

# Getting Started

The fastest way to understand CompNeuroVis is to run one of the example apps.

## Install

Base install:

```bash
pip install -e .
```

If you want local docs authoring or PR-readiness checks:

```bash
pip install -e ".[contrib]"
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

### Surface Plus Linked Cross-Section

Run:

```bash
python examples/surface_plot/surface_cross_section_visualizer.py
```

Use this if you want a 3-D surface linked to a line plot through shared controls.
It uses a reusable `GridSliceOperatorSpec` so the slice logic is not owned by the surface view itself.

### Live NEURON Morphology Viewer

Run:

```bash
python examples/neuron/visualizer_example.py
```

Requires:

```bash
pip install -e ".[neuron]"
```

Use this if you want a live SWC-backed morphology view with traces.

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

### Replay a Precomputed Animation

Run:

```bash
python examples/surface_plot/animated_surface_replay.py
```

Use this if your data already exists as frames and you want to play them back through the same frontend.

## Next Steps

- Read [Build a static surface](tutorials/build-a-static-surface.md) if you want to adapt the static surface example.
- Read [Build a NEURON session](tutorials/build-a-neuron-session.md) if you want to create a live backend by subclassing `NeuronSession`.
- Read [Build a Jaxley session](tutorials/build-a-jaxley-session.md) if you want to subclass `JaxleySession`.
- Read [Build a replay app](tutorials/build-a-replay-app.md) if you already have computed frames.
- Browse the generated [Example Index](reference/example-index.md) for the current runnable examples.

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
