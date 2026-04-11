---
title: Release Process
summary: Lightweight release workflow for version bumps, changelog updates, sealed branch tips, tags, and GitHub Releases.
---

# Release Process

CompNeuroVis keeps release mechanics intentionally simple:

- `pyproject.toml` stores the package version
- `CHANGELOG.md` stores the canonical human-readable release history
- a git tag such as `v0.2.0` marks the immutable release point
- a GitHub Release publishes that tagged version

## Working Model

Treat `CHANGELOG.md` as the source of truth for release notes.

Use GitHub Releases as the public wrapper around a tagged commit, not as the place where release history is first written down.

## Normal Flow

1. Keep new notable work under `## Unreleased` in `CHANGELOG.md`.
2. When preparing a release, move those notes into a new version section such as `## 0.2.0 - 2026-04-11`.
3. Update `pyproject.toml` to the same version.
4. Run `python scripts/pr_readiness.py check`.
5. Commit the release-prep changes normally.
6. Run `python scripts/pr_readiness.py seal --commit` so the branch tip is the explicit final attestation commit.
7. Tag that sealed commit with `git tag v0.2.0`.
8. Push the branch and the tag.
9. Create a GitHub Release from the tag and reuse the matching `CHANGELOG.md` section as the release notes.

## Practical Rules

- Tag the sealed final commit, not the implementation commit that precedes it.
- If anything changes after sealing, rerun the checks, create a new seal commit, and retag the new final commit.
- Keep changelog entries short and user-facing: focus on behavior, workflows, packaging, and docs that matter to users.
- Do not use `docs/architecture/refactor-tracker.md` as a substitute for release notes; it is a design tracker, not a public changelog.

## Suggested Changelog Shape

Use a small stable structure:

- `Changed`
- `Added`
- `Fixed`
- `Docs`

That is enough for the current scale of the repo without adding release-fragment tooling.
