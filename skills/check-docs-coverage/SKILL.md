---
name: check-docs-coverage
description: Audit whether every name exported in compneurovis.__all__ appears in at least one authored doc. Use in the pr-readiness pipeline whenever public API is added or changed, and whenever a suspected doc gap needs surfacing.
kind: coverage
surface: docs
stage: verify
trust: general
---

# Check Docs Coverage

Run this whenever the public API changes or you suspect doc gaps.

## What counts as authored doc coverage

A name is considered covered if it appears in any `.md` file under `docs/` **except** `docs/reference/api-index.md` and `docs/reference/repo-map.md` (both generated and do not count).

## Steps

1. Read `src/compneurovis/__init__.py` and collect every name in `__all__`, including those conditionally added by optional backend try/except blocks.

2. For each name, search `docs/` (excluding `api-index.md` and `repo-map.md`) for any mention.

3. Classify each name as one of:
   - **Covered** — appears in at least one authored doc.
   - **Uncovered** — appears only in generated files or not at all. Requires action.

4. For each **Uncovered** name, also check whether it appears in any file under `examples/` or `src/`. This determines the right action:
   - Used in examples or source but undocumented → **add authored doc coverage** (tutorial, architecture note, or concept doc).
   - Not used anywhere outside its definition and `__init__.py` → **consider removing from `__all__`** (likely prematurely exported).

5. Report results grouped by classification. For each uncovered name state: what it is (one sentence from reading its definition), where it is used, and the recommended action.

## Passing threshold

All names in `__all__` must be covered before a change is considered doc-complete. Uncovered names are required work, not skippable, unless explicitly recorded as named deferred gaps with a reason in the readiness report.

## What not to do

- Do not count `docs/reference/api-index.md` or `docs/reference/repo-map.md` as coverage.
- Do not create stub docs that only repeat the name without explaining purpose or usage.
