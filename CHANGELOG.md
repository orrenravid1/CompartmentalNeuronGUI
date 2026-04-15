# Changelog

This changelog is the canonical human-readable release history for CompNeuroVis.

Use it together with:

- `pyproject.toml` for the package version
- Git tags such as `v0.2.0` for immutable release points
- GitHub Releases for published release notes tied to a tag

## Unreleased

### Added

- Added a GitHub Pages workflow that validates docs on pull requests and deploys the strict MkDocs site from `main`.

### Docs

- Documented the GitHub Pages publishing path and required one-time repo Pages configuration.

## 0.2.0 - 2026-04-11

### Changed

- Refactored the core model around `Field`, `Scene`, typed updates, and optional live/replay `Session`s.
- Consolidated backend-backed workflows around shared builders and a common frontend/session architecture.
- Standardized terminology around `Scene` and `startup_scene(...)` across the docs and current public model.

### Added

- Added a sealed PR-readiness attestation workflow with a final commit receipt under `.compneurovis/pr-readiness/`.
- Added a strict MkDocs + Material + `mkdocstrings` docs site for authored guides plus generated API reference.
- Added packaging metadata validation for Poetry extras, lockfile consistency, and contributor tooling install surfaces.
- Added a named contributor extra so local docs/test tooling can be installed with `pip install -e ".[contrib]"`.

### Docs

- Reworked the README toward a user-first introduction with clearer example entrypoints and contributor guidance.
- Added stricter docs validation for markdown paths, docs vocabulary drift, and docs-site build health.
