# CompNeuroVis

CompNeuroVis is a PyQt6/VisPy visualization toolkit for compartmental neuroscience workflows. The current architecture is centered on:

- `Field`: dense labeled data with named axes and coordinates
- `Document`: static fields, geometries, views, controls, and layout
- `Session`: optional live or replay backend that emits typed updates
- `PipeTransport`: local process transport for Python/VisPy workflows
- `VispyFrontend`: the current frontend implementation

Conceptually, the architecture is composed along three independent axes:

- backend/runtime, such as NEURON, Jaxley, or replay
- features, such as traces, morphology, surfaces, controls, and actions
- layout, such as plot-only, mixed 3D + plot, and future workbench arrangements

These axes are meant to stay orthogonal. A NEURON-backed app is not automatically a morphology app. For example, a signaling-cascade viewer can be a NEURON-backed live app with controls and traces but no morphology.

## Install

```bash
pip install -e .
```

Optional simulator backends are exposed as Poetry extras:

```bash
pip install -e .[neuron]
```

## Quick Start

Static surface:

```bash
python examples/static_surface_visualizer.py
```

Surface cross section:

```bash
python examples/surface_cross_section_visualizer.py
```

Live NEURON morphology:

```bash
python examples/neuron/visualizer_example.py
```

On Windows, live session entrypoints use `multiprocessing` with `spawn`. `run_app(...)` ignores spawned child imports internally so a shared script can keep the same top-level launch pattern across Windows, Linux, and macOS. A manual `if __name__ == "__main__":` wrapper is optional, not a library requirement.

## Public API

The main package exports the core types, builders, and frontend entrypoint:

- `Field`, `Document`, `MorphologyGeometry`, `GridGeometry`
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
