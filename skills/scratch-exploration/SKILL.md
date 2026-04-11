---
name: scratch-exploration
description: Write a one-off exploratory script, prototype, or technical spike. Use when testing a library, validating an assumption, or prototyping something before committing to integration. Do NOT use for polished examples, reusable utilities, or tests.
kind: authoring
surface: examples
stage: explore
trust: general
---

# Scratch Exploration

Use the `scratch/` folder for any script that is exploratory, throwaway, or not yet ready
to be an example. Read `scratch/README.md` first for the full distinction.

## Rules

- Place the file directly under `scratch/` (no subdirectories needed).
- Name it descriptively: `scratch/<topic>_exploration.py` or `scratch/<topic>_spike.py`.
- Add a top-of-file docstring explaining what you are testing and why.
- **Do not** run `scripts/generate_indexes.py` — scratch files are intentionally excluded
  from the docs index.
- **Do not** add imports from `compneurovis` unless you are actually testing the package.
  Pure library explorations (e.g. vispy, networkx) should stand alone.

## When to graduate to examples/

Once the exploration validates an approach and you want to show it to users, follow the
`add-example` skill instead. Polished examples live under `examples/`, are indexed, and
must follow the style conventions described in `skills/add-example/SKILL.md`.
