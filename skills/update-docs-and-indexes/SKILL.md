---
name: update-docs-and-indexes
description: Update CompNeuroVis documentation, AGENTS metadata, package READMEs, and generated reference indexes after code or workflow changes. Use when implementation is finished or when documentation has drifted from the current architecture.
---

# Update Docs and Indexes

Read `AGENTS.md` first.

Update documentation in this order:

1. Update the closest package `README.md` files for touched packages.
2. Update `AGENTS.md` if the package map, public API, extension points, commands, or skill catalog changed.
3. Update authored docs in `docs/architecture`, `docs/concepts`, or `docs/tutorials` when the underlying behavior or recommended workflow changed.
4. Update or add relevant skills under `skills/` if the workflow changed in a reusable way.
5. Regenerate the reference docs with `python scripts/generate_indexes.py`.
6. Validate with:
   - `python scripts/generate_indexes.py --check`
   - `python -m mkdocs build --strict`
   - targeted tests or `pytest` if docs describe runnable behavior

Keep docs concise, cross-reference canonical pages instead of duplicating long explanations, and prefer stable package entrypoints over low-level file internals.

