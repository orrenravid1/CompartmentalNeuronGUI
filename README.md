# CompNeuroVis

CompNeuroVis is a desktop visualization toolkit for computational neuroscience. It is meant for the common lab problem of getting arrays, traces, and compartment morphologies on screen quickly without building a custom GUI from scratch.

Use it to:

- render a 2-D field as an interactive 3-D surface
- link a surface view to a line-plot slice and controls
- view live NEURON or Jaxley compartment activity on morphology and traces
- replay precomputed frames through the same frontend and layout system

If you just want to see something working with no simulator backend, start with the static surface example after install:

```bash
pip install -e .
python examples/surface_plot/static_surface_visualizer.py
```

The current frontend is a local PyQt6/VisPy desktop app, so examples should be run in a normal GUI session.

## Install

Base install:

```bash
pip install -e .
```

If you plan to contribute, build the docs site locally, or run the PR-readiness checks, also install the docs/test toolchain:

```bash
pip install -e . pytest mkdocs mkdocs-material "mkdocstrings[python]"
```

Optional simulator backends:

```bash
pip install -e .[neuron]
```

```bash
pip install -e .[jaxley]
```

## Start Here

Choose the example that matches what you want to do first:

- First look, no simulator required:

```bash
python examples/surface_plot/static_surface_visualizer.py
```

Interactive 3-D sinc surface with appearance controls.

- Surface plus linked cross-section plot:

```bash
python examples/surface_plot/surface_cross_section_visualizer.py
```

Static field with a moveable slice plane and matching line plot.

- Live NEURON morphology viewer:

```bash
python examples/neuron/visualizer_example.py
```

Single-cell live session loaded from an SWC morphology. Requires `pip install -e .[neuron]`.

- Live Jaxley multicell example:

```bash
python examples/jaxley/multicell_example.py
```

Three procedurally built cells with synaptic connectivity. Requires `pip install -e .[jaxley]`.

- Replay a precomputed animation:

```bash
python examples/surface_plot/animated_surface_replay.py
```

Precomputed frames played through the same frontend model as live sessions.

On Windows, live session entrypoints use `multiprocessing` with `spawn`. `run_app(...)` handles the spawned-child import case internally so shared scripts can keep the same top-level launch pattern across Windows, Linux, and macOS.

## Learn the Toolkit

If you want to adapt an example rather than just run it:

- [Build a static surface](docs/tutorials/build-a-static-surface.md)
- [Build a NEURON session](docs/tutorials/build-a-neuron-session.md)
- [Build a Jaxley session](docs/tutorials/build-a-jaxley-session.md)
- [Build a replay app](docs/tutorials/build-a-replay-app.md)
- [Browse all runnable examples](docs/reference/example-index.md)

## Local Docs Site

Serve the docs site locally:

```bash
mkdocs serve
```

Build the docs site in strict mode:

```bash
mkdocs build --strict
```

## Mental Model

You do not need the full architecture to run an example, but the core model is small:

- `Field`: labeled numeric data with named axes and coordinates
- `Scene`: the fields, geometries, views, controls, and layout shown in the app
- `Session`: an optional live or replay backend that emits updates over time
- `run_app(...)`: launches the current VisPy frontend for an `AppSpec`

## Public API

The main package exports the core types, builders, and frontend entrypoint:

- `Field`, `Scene`, `MorphologyGeometry`, `GridGeometry`
- `MorphologyViewSpec`, `SurfaceViewSpec`, `LinePlotViewSpec`
- `NeuronSession`
- `build_neuron_app`, `build_surface_app`, `build_replay_app`
- `run_app`

`build_neuron_app(...)` is a current convenience helper for NEURON-backed workflows, not the intended long-term conceptual boundary of the library. The long-term direction is a feature-composable public API over the same shared core model.

## Repository Docs

- `AGENTS.md` is the canonical machine-readable entrypoint for humans and tools
- `docs/architecture/` contains architecture and protocol notes
- `docs/concepts/` contains stable conceptual primitives
- `docs/tutorials/` contains builder- and use-case-oriented walkthroughs
- `docs/reference/` contains generated repo/API/example/skill indexes
- `skills/` contains repo-owned task skills

Generate the reference indexes with:

```bash
python scripts/generate_indexes.py
```

## Contributor PR Flow

Human contributors can use the same PR-readiness checks without an agent:

1. Install contributor dependencies with `pip install -e . pytest mkdocs mkdocs-material "mkdocstrings[python]"`.
2. Run `python scripts/pr_readiness.py check` while iterating locally. This runs the repo quality gate, including `pytest`, compile checks, architecture invariants, and generated index validation.
3. Commit your implementation changes normally.
4. As the last commit before you push or open the PR, run `python scripts/pr_readiness.py seal --commit`.

`seal --commit` does not replace your normal code or docs commits. It reruns the readiness checks, writes a commit-keyed receipt under `.compneurovis/pr-readiness/`, and adds one final attestation commit automatically.

If you change code, docs, tests, examples, or skills after sealing, run `python scripts/pr_readiness.py seal --commit` again from the new final implementation commit.
