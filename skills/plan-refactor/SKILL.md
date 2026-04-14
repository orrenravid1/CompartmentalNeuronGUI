---
name: plan-refactor
description: Plan a multi-file CompNeuroVis refactor by mapping all touch points before executing. Use when a rename, protocol contract change, layout model change, or structural edit spans more than 3 files and execution order matters.
metadata:
  kind: coverage
  surface: cross-cutting
  stage: verify
  trust: general
---

# Plan Refactor

Read `AGENTS.md` first, especially the package map and non-obvious invariants.

## Steps

### 1. State the target precisely

Write one sentence: what type, name, field, method, or contract is changing and what it is changing to. Ambiguity here propagates into every downstream step.

### 2. Map touch points

Scan these surfaces in order:

- **`src/`** - definition site, all call sites, re-exports in `__init__.py`
- **`tests/`** - assertions that reference the old name or behavior
- **`examples/`** - imports and usage patterns
- **`docs/`** - authored docs, concept docs, tutorials, architecture docs that mention the name or describe the behavior
- **`skills/`** - any skill whose instructions reference the old pattern
- **Generated indexes** - `docs/reference/api-index.md`, `docs/reference/repo-map.md`, `docs/reference/example-index.md`
- **`AGENTS.md`** - package map, public API map, non-obvious invariants
- **`docs/architecture/invariants.json`** - banned terms or enforced vocabulary

Classify each touch as:
- `must change` - broken or wrong without the update
- `should update` - not broken but misleading or stale
- `verify only` - confirm behavior is still correct, no edit expected

### 3. Order execution steps

Sequence edits to minimize broken intermediate states:

- If renaming a public symbol: update the definition and re-export first, then callers, then docs and indexes last.
- If changing a protocol message: update the emitter and receiver together in one commit before touching builders or examples.
- If changing a `LayoutSpec` field: update `scene.py` first, then `frontend.py` consumers, then builders, then examples.
- Always regenerate indexes after the last code/doc change, not before.

### 4. Identify verification checkpoints

After each logical group of changes, state which commands to run:

- `python -m compileall src examples tests`
- `pytest`
- `python scripts/check_architecture_invariants.py`
- `python scripts/generate_indexes.py --check`

### 5. Flag follow-on skills

After producing the plan, note which other skills should run as follow-through:
- `check-change-impact` if the scope is larger than initially apparent
- `update-docs-and-indexes` if authored docs need editing
- `audit-skill-coverage` if the refactor introduces a new pattern worth capturing
- `breaking-rename-sweep` if the change retires a term that should be banned from the codebase going forward

## Output format

Return a numbered execution plan. Each step: what to change, which file(s), classified as `must` / `should` / `verify`, and which checkpoint follows the step group.
