---
name: update-docs-and-indexes
description: Update CompNeuroVis documentation, AGENTS metadata, package READMEs, and generated reference indexes after code or workflow changes. Mandatory when touching authored docs, AGENTS metadata, package READMEs, or generated indexes, even if those edits are incidental to another change.
metadata:
  kind: meta
  surface: docs
  stage: implement
  trust: general
---

# Update Docs and Indexes

Read `AGENTS.md` first.

Use this skill whenever you:

- edit authored docs under `docs/`
- edit `AGENTS.md`
- edit a package-local `README.md`
- regenerate or validate reference indexes
- discover that code or terminology changes have made docs stale, even if you reached that point through another skill first

If the touched docs describe current architecture or capability boundaries, also
use `audit-architecture-doc-consistency` before final validation.

Update documentation in this order:

1. Update the closest package `README.md` files for touched packages.
2. Update `AGENTS.md` if the package map, public API, extension points, commands, or skill catalog changed.
3. Update authored docs in `docs/architecture`, `docs/concepts`, or `docs/tutorials` when the underlying behavior or recommended workflow changed.
4. Audit architecture-facing prose with `audit-architecture-doc-consistency` when the edits make claims about current support, limitations, or future work.
5. Update or add relevant skills under `skills/` if the workflow changed in a reusable way.
6. Regenerate the reference docs with `python scripts/generate_indexes.py`.
7. Validate with:
   - `python scripts/generate_indexes.py --check`
   - `pytest tests/test_docs_and_indexes.py`
   - `python -m mkdocs build --strict`
   - targeted tests or `pytest` if docs describe runnable behavior

Keep docs concise, cross-reference canonical pages instead of duplicating long explanations, and prefer stable package entrypoints over low-level file internals.

