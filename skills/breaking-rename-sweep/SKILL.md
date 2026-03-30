---
name: breaking-rename-sweep
description: Apply a deliberate breaking rename or taxonomy cleanup across CompNeuroVis code, docs, generated references, and skills, then verify no banned legacy terms remain. Use when a canonical architectural term changes and compatibility aliases are not desired.
---

# Breaking Rename Sweep

Read `AGENTS.md`, `docs/architecture/refactor-tracker.md`, and `docs/architecture/invariants.json` first.

Use this when the repo has decided on a new canonical term and wants to remove the old one completely.

1. Update the canonical name everywhere it matters:
   - `src/`
   - `tests/`
   - `docs/`
   - `skills/`
   - `AGENTS.md`
2. Remove compatibility aliases and legacy exports unless the user explicitly requests a staged migration.
3. Update `docs/architecture/invariants.json` so the retired names are banned by automation.
4. Regenerate derived docs with `python scripts/generate_indexes.py`.
5. Validate the rename with:
   - `python scripts/check_architecture_invariants.py`
   - `pytest`
   - `python -m compileall src examples tests`

Keep the vocabulary strict:

- `Replace` means full replacement.
- `Patch` means partial in-place change.
- `Append` means ordered extension.

If a rename introduces a reusable maintenance workflow, follow with `register-skill`.
