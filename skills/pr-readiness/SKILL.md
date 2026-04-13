---
name: pr-readiness
description: Run the CompNeuroVis pre-PR maintenance workflow by checking downstream change impact, test drift, docs/index updates, and skill registration updates before a branch is considered ready for review. Use near the end of a change when Codex should apply the relevant repo maintenance skills in sequence.
kind: orchestration
surface: cross-cutting
stage: release
trust: general
---

# PR Readiness

Use this as an orchestration skill near the end of a change. It does not replace the narrower skills; it tells you which ones to apply and in what order.

Run this sequence:

1. Use `check-change-impact` to determine which docs, tests, examples, exports, and skills are affected.
2. Use `check-docs-coverage` to verify every name in `__all__` appears in at least one authored doc. Uncovered names are required work before the change is doc-complete.
3. If the change added a builder, session type, view spec, or geometry type, use `check-tutorial-coverage` to verify each has a tutorial or non-debug example.
4. If the change introduced a new major primitive or interaction model, use `check-concept-coverage` to verify a concept doc exists for it.
5. Use `check-test-coverage-drift` to confirm the current tests still defend the changed behavior.
6. Use `audit-skill-coverage` to determine whether the change introduces a new reusable workflow that warrants a new skill.
7. If the change is a deliberate terminology or taxonomy rename, use `breaking-rename-sweep`.
8. If a skill was created or renamed, use `register-skill`.
9. Use `update-docs-and-indexes` when code paths, package boundaries, public API, workflow docs, `AGENTS.md`, package READMEs, or generated indexes changed. This is still required when the docs edits were incidental to another change.
10. Run the verification set that matches the impact:
   - targeted `pytest` modules first
   - `python scripts/check_architecture_invariants.py`
   - `python scripts/check_packaging_metadata.py`
   - `pytest`
   - `python -m compileall src examples tests`
   - `python scripts/generate_indexes.py --check`
   - `python -m mkdocs build --strict`
   - optionally, for stricter docs-language CI, `python scripts/check_docs_vocabulary.py --fail-on-warnings`
   - for human contributors without agent support, `python scripts/pr_readiness.py check` is the one-command local quality gate
11. When the implementation commit is complete and the working tree is clean, run `python scripts/pr_readiness.py seal`.
12. Human contributors should usually prefer `python scripts/pr_readiness.py seal --commit`, which creates the standalone final attestation commit automatically.
13. If any code, docs, examples, or skill files change after the seal, regenerate the attestation and create a new final seal commit.

Finish with a short readiness report that states:

- what changed
- which maintenance skills were applied
- what verification ran
- which commit was sealed by `python scripts/pr_readiness.py seal`
- any remaining risk, skipped coverage, or manual GUI checks still needed

Do not mark a change ready if tests or generated indexes are knowingly stale.
