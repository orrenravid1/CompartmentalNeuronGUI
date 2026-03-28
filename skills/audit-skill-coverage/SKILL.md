---
name: audit-skill-coverage
description: Check whether a CompNeuroVis change introduces a new reusable workflow that deserves a repo-owned skill. Use after check-change-impact and before register-skill in the pr-readiness pipeline.
---

# Audit Skill Coverage

Run this after `check-change-impact` has identified what the change introduced.

1. List the new workflows, patterns, or primitives the change added. Ignore pure bug fixes and doc-only changes.
2. Check `skills/` and `AGENTS.md` to see if any existing skill already covers the new workflow at a useful level of specificity.
3. Apply this decision rule for each uncovered workflow:

   **Create a skill if:**
   - A future agent or human would need to repeat this workflow
   - The workflow is non-trivial enough that it is easy to do wrong without guidance
   - The workflow is not already covered by a more general existing skill

   **Do not create a skill if:**
   - The workflow is a minor variant already handled by an existing skill
   - It is a one-off with no plausible future reuse
   - The steps are already fully described in a canonical doc that is easy to find

4. For each workflow that warrants a skill: either create the skill now (then use `register-skill`) or record it explicitly in the readiness report as a known gap.
5. Report the outcome — for each new workflow, state whether a skill was created, already existed, or was intentionally deferred and why.

Do not create a skill for every changed file. The bar is reusability and non-triviality, not completeness.
