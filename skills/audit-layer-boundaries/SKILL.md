---
name: audit-layer-boundaries
description: Check the CompNeuroVis import graph for structural layer violations across core, session, builders/backends, and frontends. Use when a refactor or new module may have introduced a cross-layer import that bypasses the intended dependency direction.
kind: coverage
surface: cross-cutting
stage: verify
trust: general
---

# Audit Layer Boundaries

Read `AGENTS.md` package map first.

The four layers and their allowed import direction:

```
core  ←  session  ←  builders / backends  ←  frontends
```

A layer may import from layers to its left. It must not import from layers to its right.

## Checks

For each layer, grep imports and confirm no upward dependency exists.

**`src/compneurovis/core/`**
Must not import from: `session`, `builders`, `backends`, `frontends`.

**`src/compneurovis/session/`**
Must not import from: `builders`, `backends`, `frontends`.

**`src/compneurovis/backends/` and `src/compneurovis/builders/`**
Must not import from: `frontends`.

**`src/compneurovis/frontends/`**
May import from all layers. No restriction.

## How to check

```bash
# example: find upward imports from core
grep -r "from compneurovis\.session\|from compneurovis\.backends\|from compneurovis\.frontends\|from compneurovis\.builders" src/compneurovis/core/
grep -r "from compneurovis\.backends\|from compneurovis\.frontends\|from compneurovis\.builders" src/compneurovis/session/
grep -r "from compneurovis\.frontends" src/compneurovis/backends/ src/compneurovis/builders/
```

## Allowed exceptions

- `src/compneurovis/__init__.py` re-exports across layers by design — skip it.
- Test files under `tests/` are exempt.
- Type-checking-only imports under `TYPE_CHECKING` guards are low-risk but worth flagging if they touch a disallowed layer.

## Reporting

For each layer: either "clean" or list the violating import with file and line. Distinguish between a genuine violation and an already-noted transitional item in the backlog.

If a violation is found: check whether it can be resolved by moving the import to a builder or by adding an abstraction in `core` that the session can depend on without reaching upward.
