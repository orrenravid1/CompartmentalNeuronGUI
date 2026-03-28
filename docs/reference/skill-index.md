---
title: Skill Index
summary: Generated list of repo-owned skills and their trigger descriptions.
---

# Skill Index

- `add-example`: Add a new runnable example to CompNeuroVis. Use when demonstrating a new workflow, builder pattern, or visualization type that is not already covered by an existing example. (`skills/add-example/SKILL.md`)
- `add-simulator-backend`: Add or update a simulator backend/session for CompNeuroVis. Use when creating a new live or replay backend under compneurovis.backends, wiring it to the Session protocol, or documenting how a backend should emit DocumentReady, FieldUpdate, and DocumentPatch updates. (`skills/add-simulator-backend/SKILL.md`)
- `add-static-field-visualization`: Add or update a static field, surface, or slice-based visualization in CompNeuroVis. Use when building new static documents, grid-backed surfaces, line-plot slices, or field-oriented example apps with the builder layer. (`skills/add-static-field-visualization/SKILL.md`)
- `add-view-panel`: Add or update a frontend view or panel in CompNeuroVis. Use when extending the VisPy frontend with a new panel, renderer, or view-spec consumer, or when changing how frontend-owned state maps onto rendered output. (`skills/add-view-panel/SKILL.md`)
- `audit-skill-coverage`: Check whether a CompNeuroVis change introduces a new reusable workflow that deserves a repo-owned skill. Use after check-change-impact and before register-skill in the pr-readiness pipeline. (`skills/audit-skill-coverage/SKILL.md`)
- `check-change-impact`: Review a CompNeuroVis change set and determine what code, tests, docs, AGENTS metadata, generated indexes, and repo-owned skills need to be updated. Use when code has changed and Codex should audit downstream impact before or after implementation. (`skills/check-change-impact/SKILL.md`)
- `check-test-coverage-drift`: Audit whether a CompNeuroVis change set is still covered by the current tests and add or update tests when behavior, protocol contracts, builders, docs validations, or examples change. Use before merging code when test expectations may have drifted from the implementation. (`skills/check-test-coverage-drift/SKILL.md`)
- `debug-protocol-dataflow`: Debug dataflow, transport, and session/frontend integration issues in CompNeuroVis. Use when DocumentReady, FieldUpdate, or DocumentPatch messages are missing, malformed, stale, or not producing the expected frontend behavior. (`skills/debug-protocol-dataflow/SKILL.md`)
- `debug-rendering`: Debug visual rendering issues in CompNeuroVis — wrong colors, missing geometry, blank panels, or performance problems. Use when the protocol dataflow is confirmed correct but the rendered output is wrong or absent. (`skills/debug-rendering/SKILL.md`)
- `pr-readiness`: Run the CompNeuroVis pre-PR maintenance workflow by checking downstream change impact, test drift, docs/index updates, and skill registration updates before a branch is considered ready for review. Use near the end of a change when Codex should apply the relevant repo maintenance skills in sequence. (`skills/pr-readiness/SKILL.md`)
- `register-skill`: Register a newly created CompNeuroVis repo skill in the canonical metadata and generated indexes. Use after adding or renaming a skill under `skills/` when `AGENTS.md` and `docs/reference/skill-index.md` must be updated and validated. (`skills/register-skill/SKILL.md`)
- `update-docs-and-indexes`: Update CompNeuroVis documentation, AGENTS metadata, package READMEs, and generated reference indexes after code or workflow changes. Use when implementation is finished or when documentation has drifted from the current architecture. (`skills/update-docs-and-indexes/SKILL.md`)
