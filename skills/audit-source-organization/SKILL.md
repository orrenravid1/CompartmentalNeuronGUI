---
name: audit-source-organization
description: Review CompNeuroVis source files for maintainability of file layout, top-level symbol order, class method order, abstraction jumps, long methods, and names-only readability. Use when assessing whether code is easy to read top-down, when a file feels scrambled or accreted, or before/after refactors that reorganize frontend, renderer, backend, session, or core implementation files.
metadata:
  kind: coverage
  surface: cross-cutting
  stage: verify
  trust: general
---

# Audit Source Organization

Use this skill to review source organization as its own quality axis. The goal is
not to judge whether behavior works, but whether a new contributor can predict
where code lives, read a file top-down, and change one behavior without loading
the whole subsystem into memory.

Read `AGENTS.md` first. If the review touches frontend rendering structure, also
read `src/compneurovis/frontends/vispy/README.md`.

## 1. Generate Names-Only Outlines

For each target file, produce a top-down outline of top-level functions, classes,
and class methods with line numbers. Use an AST-based command so comments and
nested implementation details do not distort the outline.

```bash
python skills/audit-source-organization/scripts/source_outline.py PATH [PATH ...]
```

When reviewing multiple files, keep outlines separate. Do not hide an awkward
file order behind a subsystem summary.

## 2. Classify Symbols

Classify each top-level symbol as one of:

- `public API`
- `primary implementation`
- `secondary helper`
- `private utility`
- `adapter/glue`
- `debug/perf`

The first screen of a source file should usually introduce public API or primary
implementation concepts. Large private utilities should not bury the main
objects unless they are required to understand every later symbol.

## 3. Review Top-Down Readability

Check whether the outline tells a coherent story by name alone:

- Does the file purpose become obvious from the first 30 lines?
- Do primary concepts appear before deep implementation details?
- Does the file move from broad concepts to local helpers, or does it bounce
  between unrelated abstraction levels?
- Can a reader predict where to find update handling, rendering, layout, state,
  transport, or math helpers?
- Are helpers near their only caller, or extracted into a clearly named module?

Call out files that read like a feature accumulation log rather than a deliberate
library surface.

## 4. Review Method Order

Within classes, prefer this order unless local conventions strongly differ:

1. `__init__`
2. public methods and properties
3. event/update entrypoints
4. main rendering/application logic
5. private helpers
6. formatting, cache, and math helpers

Flag classes where a reader must jump hundreds of lines between state setup,
entrypoints, dirty-state mutation, and the methods that consume that state.

## 5. Review Method Size and Phase Mixing

Flag methods that are too large or mix phases. As a rule of thumb:

- 40-70 lines: inspect closely
- more than 70 lines: likely needs named phases or extraction
- more than 120 lines: almost always a maintainability problem

Look for methods that combine parsing, validation, scene mutation, refresh
planning, rendering, error handling, and logging. A long method is less serious
when it is one linear algorithm with clear phases and local variables.

## 6. Review Duplication of Navigation Logic

Search for duplicated routines that answer structural questions such as:

- which panel owns a view
- which fields affect a view
- which operators belong to a view
- which update targets should refresh
- which state keys are bound by a view

Duplicated navigation logic is an organization smell even if both copies are
currently correct.

## 7. Score Each File

Score each item from 0 to 2:

- File purpose obvious from first 30 lines
- Top-level symbol order tells a coherent story
- Main concepts appear before deep implementation details
- Function names form a readable outline
- Method order follows lifecycle / public API / helpers
- No method mixes more than one responsibility level
- No duplicated navigation or classification logic
- Private helpers are near their only caller or extracted

Interpret the total:

- `14-16`: good
- `10-13`: acceptable but drifting
- `6-9`: hard to maintain
- `0-5`: actively obscuring the design

## Reporting

Report findings with file and line number. Lead with the worst read-order or
organization problems, then include the score table. For each finding, state why
the current organization slows comprehension and what concrete move would make
the file easier to read.

Do not propose broad renames or module splits without naming the new boundaries.
Prefer small, reviewable reorganizations such as extracting refresh scheduling,
update reduction, colormap helpers, overlays, or shared backend history helpers.
