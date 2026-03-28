---
name: pr-readiness
description: Run the CompNeuroVis pre-PR maintenance workflow by checking downstream change impact, test drift, docs/index updates, and skill registration updates before a branch is considered ready for review. Use near the end of a change when Codex should apply the relevant repo maintenance skills in sequence.
---

# PR Readiness

Use this as an orchestration skill near the end of a change. It does not replace the narrower skills; it tells you which ones to apply and in what order.

Run this sequence:

1. Use `check-change-impact` to determine which docs, tests, examples, exports, and skills are affected.
2. Use `check-test-coverage-drift` to confirm the current tests still defend the changed behavior.
3. Use `audit-skill-coverage` to determine whether the change introduces a new reusable workflow that warrants a new skill.
4. If a skill was created or renamed, use `register-skill`.
5. Use `update-docs-and-indexes` when code paths, package boundaries, public API, or workflow docs changed.
5. Run the verification set that matches the impact:
   - targeted `pytest` modules first
   - `pytest`
   - `python -m compileall src examples tests`
   - `python scripts/generate_indexes.py --check`

Finish with a short readiness report that states:

- what changed
- which maintenance skills were applied
- what verification ran
- any remaining risk, skipped coverage, or manual GUI checks still needed

Do not mark a change ready if tests or generated indexes are knowingly stale.
