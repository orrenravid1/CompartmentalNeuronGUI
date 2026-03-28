---
name: add-simulator-backend
description: Add or update a simulator backend/session for CompNeuroVis. Use when creating a new live or replay backend under compneurovis.backends, wiring it to the Session protocol, or documenting how a backend should emit DocumentReady, FieldUpdate, and DocumentPatch updates.
---

# Add a Simulator Backend

Read `AGENTS.md`, `docs/architecture/core-model.md`, and `docs/architecture/session-protocol.md` first.

Implement new backends under `src/compneurovis/backends/<name>`.

Follow these rules:

- Subclass `Session` or `BufferedSession`.
- Emit typed protocol updates; do not call frontend code directly.
- Build static structure as a `Document` and emit it via `DocumentReady`.
- Express measured/simulated values as `Field` objects or `FieldUpdate`s.
- Keep GUI state out of the backend. Use `SetControl` and `InvokeAction` as semantic inputs.

Update:

- the backend package `README.md`
- `AGENTS.md` package map or extension points if the public surface changes
- generated indexes with `python scripts/generate_indexes.py`

