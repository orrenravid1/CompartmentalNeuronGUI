---
name: audit-architecture-doc-consistency
description: Audit CompNeuroVis architecture, concept, backlog, roadmap, and proposal docs for stale, understated, overstated, or contradictory capability claims against the current repo behavior. Use after editing architecture-facing docs, when writing a design proposal, or when a sentence about current support may no longer match code, examples, or tests.
metadata:
  kind: coverage
  surface: docs
  stage: verify
  trust: general
---

# Audit Architecture Doc Consistency

Read `AGENTS.md` first.

Use this skill when prose makes claims about what the architecture does today,
what remains missing, or what is only proposed.

Audit in this order:

1. Scope the claim surface.
   - Start from the touched docs under `docs/architecture/`, `docs/concepts/`,
     `docs/tutorials/`, `AGENTS.md`, or proposal docs.
   - Extract the capability statements, limitation statements, and future-work
     statements that could drift.

2. Search for nearby claims.
   - Grep related docs for the same capability words and neighboring terms.
   - Include backlog, roadmap, decisions, concept docs, package `README.md`
     files, and relevant skill docs when they describe the same behavior.

3. Verify against repo truth, not only other prose.
   - Read the current code seams, examples, and tests that define the behavior.
   - For layout claims, verify `LayoutSpec`, frontend panel construction,
     example usage, and targeted frontend tests before accepting the prose.
   - Prefer direct evidence from `src/`, `examples/`, and tests over inherited
     wording from an older doc.

4. Classify each claim.
   - Correct current-state claim.
   - Understated current support.
   - Overstated current support.
   - Future-work item that is already implemented in some form.
   - Real missing capability, but phrased too broadly or imprecisely.
   - Proposal text that accidentally describes current reality as future work.

5. Reconcile the wording.
   - Say exactly what exists today.
   - Then say exactly what limitation remains.
   - Distinguish "supported today, but with transitional limits" from "not
     supported yet."
   - Do not let backlog/proposal text erase currently working capabilities.

6. Report findings with evidence.
   - For each stale claim, give file and line, stale wording, current evidence,
     and recommended replacement.
   - Separate hard contradictions from softer imprecision.

Use these severities:

- High: doc tells contributors a capability does not exist when it does, or
  says it exists when it does not.
- Medium: doc blurs current support with future direction, likely causing wrong
  design decisions or proposal framing.
- Low: wording is technically defensible but imprecise enough to invite drift.

Decision rule:

- "Implemented, but limited" must not be written as "future work may need."
- "Possible in architecture" must not be written as "shipped behavior" without
  code or example evidence.
- If the truth depends on scope, name the scope explicitly: per host, per view,
  per app, or proposal only.

Follow-through:

- If you edit docs, use `update-docs-and-indexes`.
- If this audit reveals reusable workflow drift, update the related skill docs.
- Validate with:
  - `python scripts/generate_indexes.py --check`
  - `pytest tests/test_docs_and_indexes.py`
  - `python -m mkdocs build --strict`

Do not stop at "wording could be better." If the prose changes architectural
understanding, fix it or report it as a concrete finding.
