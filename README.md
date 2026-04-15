# CompNeuroVis

CompNeuroVis is a desktop visualization toolkit for computational neuroscience. It is meant for the common lab problem of getting arrays, traces, and compartment morphologies on screen quickly without building a custom GUI from scratch.

Use it to:

- render a 2-D field as an interactive 3-D surface
- link a surface view to a line-plot slice and controls
- view live NEURON or Jaxley compartment activity on morphology and traces
- replay precomputed frames through the same frontend and layout system

Start with [CompNeuroVis Docs](https://orrenravid1.github.io/CompNeuroVis/)
and especially
[Getting Started](https://orrenravid1.github.io/CompNeuroVis/getting-started/).
If you only want the fastest local first run, install the package and launch
the static surface example:

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
pip install -e ".[contrib]"
```

Optional simulator backends:

```bash
pip install -e ".[neuron]"
```

```bash
pip install -e ".[jaxley]"
```

Extras can be combined freely. To install multiple backends, or a backend with contributor tooling:

```bash
pip install -e ".[neuron,jaxley]"
pip install -e ".[contrib,jaxley]"
pip install -e ".[all]"
```

## First Runs

Choose the path that matches what you want to do first:

- `python examples/surface_plot/static_surface_visualizer.py` for the lowest-friction first look with no simulator backend.
- `python examples/surface_plot/surface_cross_section_visualizer.py` for a linked surface and line-slice workflow.
- `python examples/neuron/visualizer_example.py` for a live SWC-backed morphology session. Requires `pip install -e ".[neuron]"`.
- `python examples/jaxley/multicell_example.py` for a live procedurally built multicell example. Requires `pip install -e ".[jaxley]"`.
- `python examples/surface_plot/animated_surface_replay.py` for replaying precomputed frames through the same frontend model.

On Windows, live session entrypoints use `multiprocessing` with `spawn`. `run_app(...)` handles the spawned-child import case internally so shared scripts can keep the same top-level launch pattern across Windows, Linux, and macOS.

## Docs Map

Use the published docs for the guided path:

- [Docs home](https://orrenravid1.github.io/CompNeuroVis/)
- [Getting started](https://orrenravid1.github.io/CompNeuroVis/getting-started/) for installation and first-run paths
- [Tutorials](https://orrenravid1.github.io/CompNeuroVis/tutorials/build-a-static-surface/) for adapting examples into your own code
- [Concept docs](https://orrenravid1.github.io/CompNeuroVis/concepts/field-model/) for the stable mental model
- [Architecture docs](https://orrenravid1.github.io/CompNeuroVis/architecture/core-model/) for implementation detail and protocol structure
- [API reference](https://orrenravid1.github.io/CompNeuroVis/api/) for the public authoring surface
- [Example index](https://orrenravid1.github.io/CompNeuroVis/reference/example-index/) for runnable examples

If you are browsing the repo directly instead of the published site, the same
material also lives under `docs/`.

## Local Docs Authoring

Serve the docs site locally:

```bash
python -m mkdocs serve
```

Build the docs site in strict mode:

```bash
python -m mkdocs build --strict
```

Published docs deploy from `.github/workflows/docs-pages.yml` to
[CompNeuroVis Docs](https://orrenravid1.github.io/CompNeuroVis/).

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
- `NeuronSession`, `JaxleySession`, `ReplaySession`
- `HistoryCaptureMode`, `grid_field`
- `build_neuron_app`, `build_jaxley_app`, `build_surface_app`, `build_replay_app`
- `run_app`

Backend-specific session classes and builders are conditional exports: install the matching backend extra first when you want to use `NeuronSession`, `JaxleySession`, `build_neuron_app(...)`, or `build_jaxley_app(...)`.

`build_neuron_app(...)` and `build_jaxley_app(...)` are current convenience helpers for backend-backed workflows, not the intended long-term conceptual boundary of the library. The long-term direction is a feature-composable public API over the same shared core model.

## Repository Docs

If you are editing the repo rather than consuming the published docs:

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

## MCP Servers

`mcp.json` at the repo root is the canonical list of MCP servers for this project. Agent-specific config files (`.claude/settings.json`, `.vscode/mcp.json`, `.cursor/mcp.json`, `.codex/config.toml`, `.gemini/settings.json`, `opencode.json`) are generated from it and committed alongside it. Do not edit them directly.

For portability, the generator may adapt local stdio launch commands in agent-specific outputs. In particular, `npx`-backed servers are wrapped so the generated configs still launch correctly on Windows.

If you add, remove, or change a server, edit `mcp.json` and regenerate:

```bash
python scripts/generate_mcp_configs.py
```

## Contributor PR Flow

Human contributors can use the same PR-readiness checks without an agent:

1. Install contributor dependencies with `pip install -e ".[contrib]"`.
2. Run `python scripts/pr_readiness.py check` while iterating locally. This runs the repo quality gate, including architecture invariants, packaging metadata validation, generated MCP config validation, `pytest`, compile checks, generated index validation, and the strict MkDocs build.
3. Commit your implementation changes normally.
4. As the last commit before you push to `main` or open a PR, run `python scripts/pr_readiness.py seal --commit`.

`seal --commit` does not replace your normal code or docs commits. It reruns the readiness checks, writes a commit-keyed receipt under `.compneurovis/pr-readiness/`, and adds one final attestation commit automatically.

If you change code, docs, tests, examples, or skills after sealing, run `python scripts/pr_readiness.py seal --commit` again from the new final implementation commit.

GitHub Actions verifies the sealed tip on pull request heads and on pushes to `main`.

## Releases And Changelog

Use [CHANGELOG.md](CHANGELOG.md) as the canonical release history and GitHub Releases as the published wrapper around a tagged commit.

Recommended release flow:

1. Keep current work under `## Unreleased` in `CHANGELOG.md`.
2. When cutting a release, move those notes into a new version section and update [pyproject.toml](pyproject.toml) to the same version.
3. Run `python scripts/pr_readiness.py check`.
4. Commit the release-prep changes.
5. Run `python scripts/pr_readiness.py seal --commit`.
6. Tag that sealed final commit, for example `git tag v0.2.0`.
7. Push the branch and tag, then create a GitHub Release from that tag using the matching changelog section as the release notes.

The detailed repo workflow lives in [docs/architecture/release-process.md](docs/architecture/release-process.md).
