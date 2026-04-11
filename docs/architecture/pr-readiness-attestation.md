---
title: PR Readiness Attestation
summary: Contributor workflow that binds PR readiness to a specific parent commit through an attestation-only final commit.
---

# PR Readiness Attestation

CompNeuroVis treats PR readiness as a versioned artifact, not just a checklist.

Use this workflow when a branch is ready for review or when a direct `main` push is ready to land:

1. Run `python scripts/pr_readiness.py check` while iterating if you want the local quality gate before committing.
2. Finish the implementation and commit the code, docs, tests, and generated files that belong to the change.
3. From a clean working tree, run `python scripts/pr_readiness.py seal --commit`.
4. Verify the branch tip with `python scripts/pr_readiness.py verify --rerun-commands` in CI or another server-side harness.

`seal --commit` is the preferred human path. It reruns the canonical checks, writes the commit-keyed JSON receipt under `.compneurovis/pr-readiness/`, and creates the standalone final attestation commit automatically.

The canonical checks currently include:

- architecture invariants
- packaging metadata validation for `pyproject.toml` and `poetry.lock`
- `pytest`
- compile checks
- generated reference index checks
- MkDocs strict-site validation

The attestation records:

- the sealed parent commit hash
- the sealed parent tree hash
- the canonical PR-readiness verification command set
- the exact subject and trailers required for the final seal commit

The verifier rejects a branch unless:

- `HEAD` has exactly one parent
- the attestation targets `HEAD^`
- the sealed tree hash matches `HEAD^`
- the final commit changes only one JSON receipt under `.compneurovis/pr-readiness/`
- the final commit subject and trailers match the attested values

This keeps the branch tip explicit. If more code or docs land after the seal, the branch is no longer PR-ready until the attestation is regenerated and recommitted as the new final commit.

The repository workflow runs this verifier on pull request head commits and on pushes to `main`, so direct-to-main work can use the same final seal model as feature branches.
