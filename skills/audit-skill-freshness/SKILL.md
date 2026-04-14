---
name: audit-skill-freshness
description: Audit repo-owned CompNeuroVis skills against the current codebase, docs, commands, and workflow conventions to find stale instructions, dead file references, outdated commands, and missing catalog wiring. Use after significant refactors, workflow changes, skill additions, or whenever a skill may no longer match current package layout, frontend/session behavior, or PR-readiness process.
metadata:
  kind: coverage
  surface: repo-infra
  stage: verify
  trust: general
---

# Audit Skill Freshness

Read `AGENTS.md`, `docs/reference/skill-index.md`, and the target skill files first.

Audit in this order:

1. Scope likely churn.
   - Review recent changes touching `src/`, `docs/`, `skills/`, `scripts/`, `AGENTS.md`, or generated indexes.
   - Prioritize skills whose instructions reference touched packages, commands, or workflows.

2. Check each targeted skill for stale references.
   - Every referenced file, script, command, and generated doc still exists.
   - Referenced APIs, classes, view/layout terms, and workflow names still exist or still mean the same thing.
   - Top-level `SKILL.md` frontmatter still matches common agent skill expectations (`name`, `description`, optional `metadata` only).
   - Repo taxonomy still exists under `metadata.kind`, `metadata.surface`, `metadata.stage`, and `metadata.trust`.
   - `SKILL.md` text remains ASCII-only so external validators and Windows-default decoders do not choke on punctuation.
   - Trigger description still matches when the skill should fire.
   - Validation commands still match `AGENTS.md` and current repo scripts.
   - Follow-through guidance still matches the current maintenance pipeline.

3. Check catalog consistency.
   - `AGENTS.md` grouped catalog includes the skill in the right bucket.
   - `docs/reference/skill-index.md` and `docs/reference/repo-map.md` are current after skill additions or renames.
   - If repo workflow changed in a reusable way, decide whether an existing skill should expand or a new skill is needed.

4. Report findings with evidence.
   - For each stale skill, give file and line, stale claim, current source of truth, and recommended fix.
   - Separate hard breakage from softer drift.

Use these severities:

- **High**: missing referenced path/script/command, renamed API, or workflow instruction that now points contributors to the wrong action.
- **Medium**: trigger wording or validation steps are materially incomplete or misleading.
- **Low**: discoverability/catalog drift or narrow wording that should expand but would not usually cause a wrong edit.

Decision rule:

- Concrete mismatch with current repo state = stale.
- Purely optional improvement with no concrete mismatch = note as enhancement, not staleness.

Follow-through:

- If you edit a skill, use `register-skill` when it was created or renamed.
- Use `update-docs-and-indexes` when `AGENTS.md`, docs, or generated indexes need updating.
- Validate with:
  - `python scripts/generate_indexes.py --check`
  - `pytest tests/test_docs_and_indexes.py`
  - `python -m mkdocs build --strict` when authored docs changed

Do not rewrite every skill just because the code changed nearby. Fix concrete drift, not style preferences.
