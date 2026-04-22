---
title: Skill Index
summary: Generated taxonomy and catalog of repo-owned skills.
---

# Skill Index

The canonical skill files stay under `skills/*/SKILL.md`. This generated index
groups them by workflow metadata so discovery does not depend on a flat path list.

## By Kind

### Authoring

- `add-control`
- `add-example`
- `add-field-visualization`
- `add-simulator-backend`
- `add-view-panel`
- `scratch-exploration`

### Coverage

- `audit-architecture-doc-consistency`
- `audit-code-smells`
- `audit-layer-boundaries`
- `audit-skill-coverage`
- `audit-skill-freshness`
- `check-change-impact`
- `check-concept-coverage`
- `check-docs-coverage`
- `check-test-coverage-drift`
- `check-tutorial-coverage`
- `plan-refactor`

### Debugging

- `debug-protocol-dataflow`
- `debug-rendering`

### Orchestration

- `pr-readiness`

### Repo Maintenance

- `breaking-rename-sweep`
- `register-skill`
- `update-docs-and-indexes`

## By Surface

### Backend

- `add-simulator-backend`

### Cross-Cutting

- `audit-code-smells`
- `audit-layer-boundaries`
- `breaking-rename-sweep`
- `check-change-impact`
- `check-test-coverage-drift`
- `debug-protocol-dataflow`
- `plan-refactor`
- `pr-readiness`

### Docs

- `audit-architecture-doc-consistency`
- `check-concept-coverage`
- `check-docs-coverage`
- `update-docs-and-indexes`

### Examples

- `add-example`
- `add-field-visualization`
- `check-tutorial-coverage`
- `scratch-exploration`

### Frontend

- `add-control`
- `add-view-panel`
- `debug-rendering`

### Repo Infrastructure

- `audit-skill-coverage`
- `audit-skill-freshness`
- `register-skill`

## By Workflow Stage

### Debug

- `debug-protocol-dataflow`
- `debug-rendering`

### Explore

- `scratch-exploration`

### Implement

- `add-control`
- `add-example`
- `add-field-visualization`
- `add-simulator-backend`
- `add-view-panel`
- `breaking-rename-sweep`
- `register-skill`
- `update-docs-and-indexes`

### Release

- `pr-readiness`

### Verify

- `audit-architecture-doc-consistency`
- `audit-code-smells`
- `audit-layer-boundaries`
- `audit-skill-coverage`
- `audit-skill-freshness`
- `check-change-impact`
- `check-concept-coverage`
- `check-docs-coverage`
- `check-test-coverage-drift`
- `check-tutorial-coverage`
- `plan-refactor`

## By Trust

### General

- `add-control`
- `add-example`
- `add-field-visualization`
- `add-simulator-backend`
- `add-view-panel`
- `audit-architecture-doc-consistency`
- `audit-code-smells`
- `audit-layer-boundaries`
- `audit-skill-coverage`
- `audit-skill-freshness`
- `check-change-impact`
- `check-concept-coverage`
- `check-docs-coverage`
- `check-test-coverage-drift`
- `check-tutorial-coverage`
- `debug-protocol-dataflow`
- `debug-rendering`
- `plan-refactor`
- `pr-readiness`
- `scratch-exploration`
- `update-docs-and-indexes`

### Maintainer Only

- `breaking-rename-sweep`

### Proposal Only

- `register-skill`

## Full Catalog

- `add-control` (kind: authoring, surface: frontend, stage: implement, trust: general): Add or update a control type in CompNeuroVis. Use when introducing a new spec class (scalar or compound) that the user can interact with in the controls panel, or when changing how a control maps onto frontend state. (`skills/add-control/SKILL.md`)
- `add-example` (kind: authoring, surface: examples, stage: implement, trust: general): Add a new runnable example to CompNeuroVis. Use when demonstrating a new workflow, builder pattern, or visualization type that is not already covered by an existing example. (`skills/add-example/SKILL.md`)
- `add-field-visualization` (kind: authoring, surface: examples, stage: implement, trust: general): Add or update a field-driven visualization in CompNeuroVis. Use when building static or precomputed field views, grid-backed surfaces, line-plot slices, or field-oriented example apps with the builder layer. (`skills/add-field-visualization/SKILL.md`)
- `add-simulator-backend` (kind: authoring, surface: backend, stage: implement, trust: general): Add or update a simulator backend/session for CompNeuroVis. Use when creating a new live or replay backend under compneurovis.backends, wiring it to the Session protocol, or documenting how a backend should emit SceneReady, FieldReplace or FieldAppend, StatePatch, Status, and ScenePatch updates. (`skills/add-simulator-backend/SKILL.md`)
- `add-view-panel` (kind: authoring, surface: frontend, stage: implement, trust: general): Add or update a frontend view or panel in CompNeuroVis. Use when extending the VisPy frontend with a new panel, renderer, or view-spec consumer, or when changing how frontend-owned state maps onto rendered output. (`skills/add-view-panel/SKILL.md`)
- `audit-architecture-doc-consistency` (kind: coverage, surface: docs, stage: verify, trust: general): Audit CompNeuroVis architecture, concept, backlog, roadmap, and proposal docs for stale, understated, overstated, or contradictory capability claims against the current repo behavior. Use after editing architecture-facing docs, when writing a design proposal, or when a sentence about current support may no longer match code, examples, or tests. (`skills/audit-architecture-doc-consistency/SKILL.md`)
- `audit-code-smells` (kind: coverage, surface: cross-cutting, stage: verify, trust: general): Detect architectural drift patterns in CompNeuroVis that pass compilation and tests but violate the intended layering model. Use after a change touches core, session, backends, or frontend code when design violations may have crept in. (`skills/audit-code-smells/SKILL.md`)
- `audit-layer-boundaries` (kind: coverage, surface: cross-cutting, stage: verify, trust: general): Check the CompNeuroVis import graph for structural layer violations across core, session, builders/backends, and frontends. Use when a refactor or new module may have introduced a cross-layer import that bypasses the intended dependency direction. (`skills/audit-layer-boundaries/SKILL.md`)
- `audit-skill-coverage` (kind: coverage, surface: repo-infra, stage: verify, trust: general): Check whether a CompNeuroVis change introduces a new reusable workflow that deserves a repo-owned skill. Use after check-change-impact and before register-skill in the pr-readiness pipeline. (`skills/audit-skill-coverage/SKILL.md`)
- `audit-skill-freshness` (kind: coverage, surface: repo-infra, stage: verify, trust: general): Audit repo-owned CompNeuroVis skills against the current codebase, docs, commands, and workflow conventions to find stale instructions, dead file references, outdated commands, and missing catalog wiring. Use after significant refactors, workflow changes, skill additions, or whenever a skill may no longer match current package layout, frontend/session behavior, or PR-readiness process. (`skills/audit-skill-freshness/SKILL.md`)
- `breaking-rename-sweep` (kind: meta, surface: cross-cutting, stage: implement, trust: maintainer-only): Apply a deliberate breaking rename or taxonomy cleanup across CompNeuroVis code, docs, generated references, and skills, then verify no banned legacy terms remain. Use when a canonical architectural term changes and compatibility aliases are not desired. (`skills/breaking-rename-sweep/SKILL.md`)
- `check-change-impact` (kind: coverage, surface: cross-cutting, stage: verify, trust: general): Review a CompNeuroVis change set and determine what code, tests, docs, AGENTS metadata, generated indexes, and repo-owned skills need to be updated. Use when code has changed and an agent should audit downstream impact before or after implementation. (`skills/check-change-impact/SKILL.md`)
- `check-concept-coverage` (kind: coverage, surface: docs, stage: verify, trust: general): Audit whether every major conceptual area in CompNeuroVis has a concept doc in docs/concepts/. Use in the pr-readiness pipeline when a new major primitive, model, or user-facing idea is introduced. (`skills/check-concept-coverage/SKILL.md`)
- `check-docs-coverage` (kind: coverage, surface: docs, stage: verify, trust: general): Audit whether every name exported in compneurovis.__all__ appears in at least one authored doc. Use in the pr-readiness pipeline whenever public API is added or changed, and whenever a suspected doc gap needs surfacing. (`skills/check-docs-coverage/SKILL.md`)
- `check-test-coverage-drift` (kind: coverage, surface: cross-cutting, stage: verify, trust: general): Audit whether a CompNeuroVis change set is still covered by the current tests and add or update tests when behavior, protocol contracts, builders, docs validations, or examples change. Use before merging code when test expectations may have drifted from the implementation. (`skills/check-test-coverage-drift/SKILL.md`)
- `check-tutorial-coverage` (kind: coverage, surface: examples, stage: verify, trust: general): Audit whether every primary user-facing API name has a tutorial or non-debug example. Use in the pr-readiness pipeline when a new builder, session type, view spec, or geometry type is added. (`skills/check-tutorial-coverage/SKILL.md`)
- `debug-protocol-dataflow` (kind: debug, surface: cross-cutting, stage: debug, trust: general): Debug dataflow, transport, and session/frontend integration issues in CompNeuroVis. Use when SceneReady, FieldReplace, FieldAppend, or ScenePatch messages are missing, malformed, stale, or not producing the expected frontend behavior. (`skills/debug-protocol-dataflow/SKILL.md`)
- `debug-rendering` (kind: debug, surface: frontend, stage: debug, trust: general): Debug visual rendering issues in CompNeuroVis - wrong colors, missing geometry, blank panels, or performance problems. Use when the protocol dataflow is confirmed correct but the rendered output is wrong or absent. (`skills/debug-rendering/SKILL.md`)
- `plan-refactor` (kind: coverage, surface: cross-cutting, stage: verify, trust: general): Plan a multi-file CompNeuroVis refactor by mapping all touch points before executing. Use when a rename, protocol contract change, layout model change, or structural edit spans more than 3 files and execution order matters. (`skills/plan-refactor/SKILL.md`)
- `pr-readiness` (kind: orchestration, surface: cross-cutting, stage: release, trust: general): Run the CompNeuroVis pre-PR maintenance workflow by checking downstream change impact, test drift, docs/index updates, and skill registration updates before a branch is considered ready for review. Use near the end of a change when an agent should apply the relevant repo maintenance skills in sequence. (`skills/pr-readiness/SKILL.md`)
- `register-skill` (kind: meta, surface: repo-infra, stage: implement, trust: proposal-only): Register a newly created CompNeuroVis repo skill in the canonical metadata and generated indexes. Use after adding or renaming a skill under `skills/` when `AGENTS.md` and `docs/reference/skill-index.md` must be updated and validated. (`skills/register-skill/SKILL.md`)
- `scratch-exploration` (kind: authoring, surface: examples, stage: explore, trust: general): Write a one-off exploratory script, prototype, or technical spike. Use when testing a library, validating an assumption, or prototyping something before committing to integration. Do NOT use for polished examples, reusable utilities, or tests. (`skills/scratch-exploration/SKILL.md`)
- `update-docs-and-indexes` (kind: meta, surface: docs, stage: implement, trust: general): Update CompNeuroVis documentation, AGENTS metadata, package READMEs, and generated reference indexes after code or workflow changes. Mandatory when touching authored docs, AGENTS metadata, package READMEs, or generated indexes, even if those edits are incidental to another change. (`skills/update-docs-and-indexes/SKILL.md`)
