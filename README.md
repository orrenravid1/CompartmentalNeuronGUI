# CompNeuroVis

CompNeuroVis is a PyQt6/VisPy visualization toolkit for compartmental neuroscience workflows. The current architecture is centered on:

- `Field`: dense labeled data with named axes and coordinates
- `Document`: static fields, geometries, views, controls, and layout
- `Session`: optional live or replay backend that emits typed updates
- `PipeTransport`: local process transport for Python/VisPy workflows
- `VispyFrontend`: the current frontend implementation

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
