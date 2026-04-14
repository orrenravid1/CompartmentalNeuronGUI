---
name: check-change-impact
description: Review a CompNeuroVis change set and determine what code, tests, docs, AGENTS metadata, generated indexes, and repo-owned skills need to be updated. Use when code has changed and an agent should audit downstream impact before or after implementation.
metadata:
  kind: coverage
  surface: cross-cutting
  stage: verify
  trust: general
---

# Check Change Impact

Read `AGENTS.md` first.

Audit changes in this order:

1. Identify the touched package or example paths.
2. Map those paths onto the package map and public API in `AGENTS.md`.
3. Decide whether the change affects:
   - exported names in `src/compneurovis/__init__.py`
   - package-local `README.md` files
   - architecture or concept docs in `docs/`
   - examples
   - skills
   - generated reference indexes
4. Run the smallest useful verification:
   - `python scripts/check_architecture_invariants.py` when public terminology, protocol names, or other architectural vocabulary changed
   - `python -m compileall src examples tests`
   - `pytest`
   - `python scripts/generate_indexes.py --check`
5. Identify call sites changed by the diff that are not covered by the test suite. `compileall` verifies syntax only; `pytest` only defends paths it exercises. Runtime failures - wrong kwargs, removed methods, type mismatches, untested code paths - are invisible to both. For each uncovered call site, either verify correctness by reading the callee's implementation, or flag it as a required manual smoke-test in the impact report.
6. Report missing follow-up edits explicitly instead of assuming docs are still correct.

When the change alters public concepts, package boundaries, or workflows, treat docs updates as required work, not optional cleanup.

If the audit says authored docs, `AGENTS.md`, package READMEs, or generated indexes need touching, the required follow-through skill is `update-docs-and-indexes`.

If the change or proposal edits architecture-facing prose about current
capabilities, limitations, or future work, also use
`audit-architecture-doc-consistency`.
