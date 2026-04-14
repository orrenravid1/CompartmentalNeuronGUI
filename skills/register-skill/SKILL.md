---
name: register-skill
description: Register a newly created CompNeuroVis repo skill in the canonical metadata and generated indexes. Use after adding or renaming a skill under `skills/` when `AGENTS.md` and `docs/reference/skill-index.md` must be updated and validated.
metadata:
  kind: meta
  surface: repo-infra
  stage: implement
  trust: proposal-only
---

# Register Skill

Read `AGENTS.md` first.

Run this workflow after creating a new repo-owned skill or changing a skill name.

1. Confirm the skill folder contains a valid `SKILL.md` with concise frontmatter and instructions.
   Required frontmatter shape:
   - top-level `name`, `description`, and `metadata`
   - repo taxonomy under `metadata.kind`, `metadata.surface`, `metadata.stage`, and `metadata.trust`
   - keep the skill name ASCII hyphen-case and keep `SKILL.md` body ASCII-only for cross-agent compatibility on Windows
2. Update the `Skill Usage` section in `AGENTS.md` if the new skill changes the grouped catalog summary or discovery guidance.
3. If the new skill changes reusable maintenance workflow, update any related meta-skills under `skills/`.
4. Regenerate the reference indexes with `python scripts/generate_indexes.py`.
5. Validate consistency with:
   - `python scripts/generate_indexes.py --check`
   - `pytest tests/test_docs_and_indexes.py`
   - Keep the top-level frontmatter compatible with common agent skill validators by storing repo taxonomy only under `metadata`.

Keep this skill narrow:

- Register the skill and its references.
- Do not rewrite unrelated architecture docs unless the new skill changes documented workflow boundaries.
- Prefer one-line catalog entries and let `docs/reference/skill-index.md` carry the generated descriptions.
