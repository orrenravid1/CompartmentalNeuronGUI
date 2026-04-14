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
- `src/compneurovis/neuronutils`: NEURON-specific SWC import, JSON round-trip, and layout helpers

Read the package-local `README.md` in those directories before making structural changes there.

## Public API Map

Primary exports live in `src/compneurovis/__init__.py`.

Key names:

- Core types: `Field`, `Scene`, `LayoutSpec`, `MorphologyGeometry`, `GridGeometry`
- Operator types: `OperatorSpec`, `GridSliceOperatorSpec`
- View types: `MorphologyViewSpec`, `SurfaceViewSpec`, `LinePlotViewSpec`
- Backend/session: `NeuronSession`, `JaxleySession`, `ReplaySession`
- Backend scene builders: `NeuronSceneBuilder`, `JaxleySceneBuilder`
- Workflow helpers: `HistoryCaptureMode`, `grid_field`
- Builders: `build_neuron_app`, `build_jaxley_app`, `build_surface_app`, `build_replay_app`
- Frontend entrypoint: `run_app`

## Extension Points

- Add new data-bearing features by introducing new `Field` instances and `ViewSpec` consumers.
- Add new reusable derived workflows by introducing new `OperatorSpec` instances and frontend consumers.
- Add new live backends by subclassing `Session` or `BufferedSession`.
- Add new transports under `src/compneurovis/session` without changing the core scene model.
- Add new frontend renderers under `src/compneurovis/frontends`.

## MCP Servers

`mcp.json` at the repo root is the canonical source of truth for MCP server
configuration. Edit it there; never edit the generated agent configs directly.
Generated outputs may adapt local stdio launch commands for portability; for
example, `npx`-backed servers are wrapped so the generated configs still work
on Windows.

Run the generator to (re)produce all agent-specific config files:

```bash
python scripts/generate_mcp_configs.py
```

Verify that generated configs are up to date (used in CI checks):

```bash
python scripts/generate_mcp_configs.py --check
```

| Server | Transport | Purpose |
|---|---|---|
| `fetch` | stdio (`npx`) | Fetch live docs from NEURON, Jaxley, VisPy, and other external sites |
| `context7` | HTTP | Version-accurate library docs for VisPy, PyQt6, NumPy, Jaxley, etc. |
| `arxiv` | stdio (`uvx`) | Search and read arXiv papers for algorithm/morphology/simulator references |
| `sequential-thinking` | stdio (`npx`) | Externalized reasoning scratchpad for multi-step planning and complex refactors |

Generated outputs (do not edit directly):

- `.claude/settings.json` — Claude Code (mcpServers key merged, other keys preserved)
- `.vscode/mcp.json` — VS Code / GitHub Copilot
- `.cursor/mcp.json` — Cursor
- `.codex/config.toml` — OpenAI Codex
- `.gemini/settings.json` — Gemini CLI (mcpServers key merged, other keys preserved)
- `opencode.json` — OpenCode (mcp key merged; different structure: merged command array, `environment` not `env`, explicit `type` field)

Not generated (no repo file):

- GitHub Copilot Cloud Agent — configured via GitHub repository Settings UI under
  `Settings > Copilot > Cloud agent > MCP configuration`. Uses the same mcpServers
  JSON schema but entered through the web interface, not a committed file.

To add or remove a server: edit `mcp.json`, rerun the generator, and commit
the canonical source plus all regenerated outputs together.

## Build, Test, and Run

- Install: `pip install -e .`
- Install contributor toolchain: `pip install -e ".[contrib]"`
- Install NEURON backend: `pip install -e ".[neuron]"`
- Install Jaxley backend: `pip install -e ".[jaxley]"`
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
- Regenerate agent MCP configs: `python scripts/generate_mcp_configs.py`
- Check agent MCP configs: `python scripts/generate_mcp_configs.py --check`
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
- The same sealed-tip verification workflow runs on pull request heads and on pushes to `main`.

## Release Flow

- Keep release notes in `CHANGELOG.md`, with current work collected under `## Unreleased`.
- When cutting a release, update both `CHANGELOG.md` and `pyproject.toml` to the same version.
- Run `python scripts/pr_readiness.py check`, commit the release-prep changes, then run `python scripts/pr_readiness.py seal --commit`.
- Tag the sealed final commit with a version tag such as `v0.2.0`.
- Create the GitHub Release from that tag and reuse the matching `CHANGELOG.md` section as the published notes.

## Skill Catalog

## Skill Usage

- Repo-local skills live under `skills/<name>/SKILL.md` and are the canonical workflow instructions for recurring repo tasks.
- Repo-local `SKILL.md` files should keep agent-compatible top-level frontmatter (`name`, `description`, optional `metadata`) and store repo taxonomy under `metadata.kind`, `metadata.surface`, `metadata.stage`, and `metadata.trust`.
- Write repo-owned skills in agent-neutral terms so the same instructions work out of the box for Codex, Gemini, Perplexity, Claude, and similar agents.
- Any agent working in this repo should consult the relevant `SKILL.md` before executing a task that matches the skill description.
- Mandatory trigger: if you touch authored docs under `docs/`, `AGENTS.md`, package-local `README.md` files, or generated reference indexes, consult `update-docs-and-indexes` even when those edits are incidental to another code change.
- Mandatory trigger: if code or terminology changes imply doc drift but the docs are not updated yet, consult `check-change-impact` first and then `update-docs-and-indexes` as required follow-through.
- Mandatory trigger: if you edit `docs/architecture/**`, `docs/concepts/**`, or design proposals and those edits make claims about current capabilities, limitations, or future work, consult `audit-architecture-doc-consistency` before finalizing the wording.
- Use `docs/reference/skill-index.md` to discover the right skill by kind, surface, workflow stage, or trust level.
- The current catalog is organized conceptually as:
  - authoring and exploration: `add-example`, `add-field-visualization`, `add-simulator-backend`, `add-view-panel`, `scratch-exploration`
  - coverage and verification: `audit-architecture-doc-consistency`, `check-change-impact`, `check-docs-coverage`, `check-tutorial-coverage`, `check-concept-coverage`, `check-test-coverage-drift`, `audit-skill-coverage`, `audit-skill-freshness`, `pr-readiness`
  - architectural quality: `audit-code-smells`, `audit-layer-boundaries`, `plan-refactor`
  - debugging: `debug-protocol-dataflow`, `debug-rendering`
  - repo maintenance: `breaking-rename-sweep`, `register-skill`, `update-docs-and-indexes`
- Read the generated skill index at `docs/reference/skill-index.md` for descriptions and canonical paths.

## Documentation Index

When recording new work, use the right file:

- **New feature idea, in-progress design, or deferred item** → `docs/architecture/design/backlog.md` with a `Phase:` tag
- **Large multi-step feature proposal or architecture investigation** → `docs/architecture/design/proposals/<topic>.md`, then add a summary link from `docs/architecture/design/backlog.md`
- **Active priority or phase milestone shift** → `docs/architecture/design/roadmap.md`
- **Settled architectural lesson or decision** → `docs/architecture/design/decisions.md` (requires deliberate review)

Files:

- Architecture overview: `docs/architecture/core-model.md`
- Roadmap (phases, next steps, transition targets): `docs/architecture/design/roadmap.md`
- Design decisions (settled doctrine and lessons): `docs/architecture/design/decisions.md`
- Backlog (deferred features, infrastructure, cleanup): `docs/architecture/design/backlog.md`
- Layout workbench proposal: `docs/architecture/design/proposals/layout-workbench-proposal.md`
- Release process: `docs/architecture/release-process.md`
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
- Changelog: `CHANGELOG.md`

## Non-Obvious Invariants and Boundaries

- Treat `Field` as the core data primitive; do not introduce new foundational "timeseries" or "surface" types when a labeled field plus a view is sufficient.
- Frontends own UI state such as selection and slice position. Backends receive semantic commands, not raw GUI events.
- `FieldReplace` replaces field values and may update coordinates; schema changes should rebuild or patch the scene explicitly.
- `ScenePatch` is intended for metadata/view/control changes, not arbitrary structural rewrites.
- Architectural vocabulary changes should be encoded in `docs/architecture/invariants.json` and enforced with `python scripts/check_architecture_invariants.py`.
- Keep docs and skills concise and cross-reference canonical docs instead of duplicating large explanations.
- Authored docs coverage is a correctness requirement for this repo, not just polish. Fresh contributors and fresh agents must be able to recover the intended model from repo state.
