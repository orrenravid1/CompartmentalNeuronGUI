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
- change package extras, install commands, or dependency requirements in a way users need to discover
- discover that code or terminology changes have made docs stale, even if you reached that point through another skill first

If the touched docs describe current architecture or capability boundaries, also
use `audit-architecture-doc-consistency` before final validation.

Update documentation in this order:

1. Update the closest package `README.md` files for touched packages.
2. Update `AGENTS.md` if the package map, public API, extension points, commands, install flows, or skill catalog changed.
3. Update top-level install docs such as `README.md`, `docs/getting-started.md`, and relevant tutorials when dependency, extra, or environment requirements changed.
4. Update authored docs in `docs/architecture`, `docs/concepts`, or `docs/tutorials` when the underlying behavior or recommended workflow changed.
5. Audit architecture-facing prose with `audit-architecture-doc-consistency` when the edits make claims about current support, limitations, or future work.
6. Update or add relevant skills under `skills/` if the workflow changed in a reusable way.
7. Regenerate the reference docs with `python scripts/generate_indexes.py`.
8. Check nav completeness: every `.md` file under `docs/` must appear in `mkdocs.yml` nav. Glob `docs/**/*.md`, then verify each path is listed. Files present on disk but absent from nav produce only an `INFO` in `mkdocs build` - they will not cause a build failure, so this must be checked explicitly.
9. Check section index pages: every nav section that contains two or more child pages must have an `index.md` at its root (e.g. `docs/tutorials/index.md`). If one is missing, create it with a short orientation paragraph and links to each child page.
10. Validate with:
    - `python scripts/generate_indexes.py --check`
    - `python scripts/check_packaging_metadata.py` when install metadata or extras changed
    - `pytest tests/test_docs_and_indexes.py`
    - `python -m mkdocs build --strict`
    - targeted tests or `pytest` if docs describe runnable behavior

Keep docs concise, cross-reference canonical pages instead of duplicating long explanations, and prefer stable package entrypoints over low-level file internals.

