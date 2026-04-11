# AGENTS

## Repo Purpose

CompNeuroVis is a neuroscience visualization toolkit with a core data model built around labeled `Field` objects, optional live/replay `Session`s, a typed update protocol, and a VisPy frontend. The current scope is Python + VisPy + local pipes; the protocol and package layout are designed so websocket transports and alternate frontends can be added later without rewriting the core model.

## Stable Package Map

- `src/compneurovis/core`: data primitives, views, controls, and app/scene types
- `src/compneurovis/session`: session base classes, protocol types, and pipe transport
- `src/compneurovis/frontends/vispy`: current frontend window, panels, and renderers
- `src/compneurovis/backends/neuron`: live NEURON session and morphology scene builder
- `src/compneurovis/backends/jaxley`: live Jaxley session and morphology scene builder
- `src/compneurovis/builders`: high-level app builders for neuron, surface, and replay workflows
- `src/compneurovis/jaxleyutils`: Jaxley-specific SWC, cache, and geometry helpers

Read the package-local `README.md` in those directories before making structural changes there.

## Public API Map

Primary exports live in `src/compneurovis/__init__.py`.

Key names:

- Core types: `Field`, `Scene`, `LayoutSpec`, `MorphologyGeometry`, `GridGeometry`
- View types: `MorphologyViewSpec`, `SurfaceViewSpec`, `LinePlotViewSpec`
- Backend/session: `NeuronSession`, `ReplaySession`
- Backend/session: `NeuronSession`, `JaxleySession`, `ReplaySession`
- Builders: `build_neuron_app`, `build_jaxley_app`, `build_surface_app`, `build_replay_app`, `grid_field`
- Frontend entrypoint: `run_app`

## Extension Points

- Add new data-bearing features by introducing new `Field` instances and `ViewSpec` consumers.
- Add new live backends by subclassing `Session` or `BufferedSession`.
- Add new transports under `src/compneurovis/session` without changing the core scene model.
- Add new frontend renderers under `src/compneurovis/frontends`.

## Build, Test, and Run

- Install: `pip install -e .`
- Compile check: `python -m compileall src examples`
- Run fast tests: `pytest`
- Run Jaxley backend tests: `pytest --run-jaxley tests/test_jaxley_backend.py`
- Check architecture invariants: `python scripts/check_architecture_invariants.py`
- Check packaging metadata: `python scripts/check_packaging_metadata.py`
- Check docs vocabulary drift: `python scripts/check_docs_vocabulary.py`
- Fail on docs vocabulary warnings in stricter CI: `python scripts/check_docs_vocabulary.py --fail-on-warnings`
- Serve docs locally: `python -m mkdocs serve`
- Build docs site: `python -m mkdocs build --strict`
- Regenerate reference indexes: `python scripts/generate_indexes.py`
- Check generated indexes: `python scripts/generate_indexes.py --check`
- Run the local PR-readiness quality gate: `python scripts/pr_readiness.py check`
- Seal PR readiness: `python scripts/pr_readiness.py seal`
- Seal PR readiness and create the final attestation commit automatically: `python scripts/pr_readiness.py seal --commit`
- Verify a sealed PR tip: `python scripts/pr_readiness.py verify`
- Verify a sealed PR tip and rerun the recorded checks: `python scripts/pr_readiness.py verify --rerun-commands`
- Run a live example: `python examples/neuron/visualizer_example.py`
- Run a static example: `python examples/surface_plot/static_surface_visualizer.py`

## PR Requirements

- Every public export in `src/compneurovis/__init__.py` must appear in at least one authored doc under `docs/`.
- Generated reference files in `docs/reference/` do not count as authored coverage.
- Required concept docs under `docs/concepts/` are a hard gate, not optional cleanup.
- The MkDocs site must build with `python -m mkdocs build --strict` before a branch is considered doc-complete.
- `pyproject.toml` and `poetry.lock` must stay in sync, and every published Poetry extra must resolve to a declared optional dependency.
- Do not mark a change PR-ready while docs coverage or concept coverage is knowingly incomplete.
- Human contributors should prefer `python scripts/pr_readiness.py check` during iteration and `python scripts/pr_readiness.py seal --commit` once the final implementation commit is in place.
- A PR-ready branch must end with a standalone attestation commit produced by `python scripts/pr_readiness.py seal`.
- That final commit must add exactly one attestation under `.compneurovis/pr-readiness/` and must target `HEAD^` via the required subject and `PR-Readiness-*` trailers.
- If any code, docs, examples, or skill content changes after sealing, regenerate the attestation and create a new final seal commit.

## Skill Catalog

## Skill Usage

- Repo-local skills live under `skills/<name>/SKILL.md` and are the canonical workflow instructions for recurring repo tasks.
- Any agent working in this repo should consult the relevant `SKILL.md` before executing a task that matches the skill description.
- If a task matches one of the cataloged skills below, read that skill first, then follow its workflow.
- Use `docs/reference/skill-index.md` to discover which skill best matches a task.

- `skills/add-example/SKILL.md`
- `skills/scratch-exploration/SKILL.md`
- `skills/add-simulator-backend/SKILL.md`
- `skills/add-field-visualization/SKILL.md`
- `skills/add-view-panel/SKILL.md`
- `skills/audit-skill-coverage/SKILL.md`
- `skills/breaking-rename-sweep/SKILL.md`
- `skills/check-change-impact/SKILL.md`
- `skills/check-concept-coverage/SKILL.md`
- `skills/check-docs-coverage/SKILL.md`
- `skills/check-test-coverage-drift/SKILL.md`
- `skills/check-tutorial-coverage/SKILL.md`
- `skills/debug-protocol-dataflow/SKILL.md`
- `skills/debug-rendering/SKILL.md`
- `skills/pr-readiness/SKILL.md`
- `skills/register-skill/SKILL.md`
- `skills/update-docs-and-indexes/SKILL.md`

Read the generated skill index at `docs/reference/skill-index.md` for descriptions.

## Documentation Index

- Architecture overview: `docs/architecture/core-model.md`
- Refactor tracker: `docs/architecture/refactor-tracker.md`
- PR readiness attestation: `docs/architecture/pr-readiness-attestation.md`
- Session protocol: `docs/architecture/session-protocol.md`
- Architecture invariants: `docs/architecture/invariants.json`
- VisPy frontend: `docs/architecture/vispy-frontend.md`
- Session/update concept: `docs/concepts/session-update-model.md`
- Field semantics: `docs/concepts/field-model.md`
- Geometry types concept: `docs/concepts/geometry-types.md`
- Controls/actions/state concept: `docs/concepts/controls-actions-state.md`
- View/layout concept: `docs/concepts/view-layout-model.md`
- Static surface tutorial: `docs/tutorials/build-a-static-surface.md`
- NEURON session tutorial: `docs/tutorials/build-a-neuron-session.md`
- Jaxley session tutorial: `docs/tutorials/build-a-jaxley-session.md`
- Replay tutorial: `docs/tutorials/build-a-replay-app.md`
- Generated repo map: `docs/reference/repo-map.md`
- Generated API index: `docs/reference/api-index.md`
- Generated example index: `docs/reference/example-index.md`

## Non-Obvious Invariants and Boundaries

- Treat `Field` as the core data primitive; do not introduce new foundational “timeseries” or “surface” types when a labeled field plus a view is sufficient.
- Frontends own UI state such as selection and slice position. Backends receive semantic commands, not raw GUI events.
- `FieldReplace` replaces field values and may update coordinates; schema changes should rebuild or patch the scene explicitly.
- `ScenePatch` is intended for metadata/view/control changes, not arbitrary structural rewrites.
- Architectural vocabulary changes should be encoded in `docs/architecture/invariants.json` and enforced with `python scripts/check_architecture_invariants.py`.
- Keep docs and skills concise and cross-reference canonical docs instead of duplicating large explanations.
- Authored docs coverage is a correctness requirement for this repo, not just polish. Fresh contributors and fresh agents must be able to recover the intended model from repo state.
