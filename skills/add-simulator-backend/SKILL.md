---
name: add-simulator-backend
description: Add or update a simulator backend/session for CompNeuroVis. Use when creating a new live or replay backend under compneurovis.backends, wiring it to the Session protocol, or documenting how a backend should emit DocumentReady, FieldReplace, and DocumentPatch updates.
---

# Add a Simulator Backend

Read `docs/architecture/core-model.md` and `docs/architecture/session-protocol.md` first.

Reference implementation: `src/compneurovis/backends/neuron/session.py`.

1. Create a package under `src/compneurovis/backends/<name>`.
2. Subclass `Session` or `BufferedSession`; emit only typed protocol updates — no direct frontend calls.
3. Emit `DocumentReady` with a fully-built `Document` on initialization.
4. Express all measured or simulated values as `Field` objects, delivered via `FieldReplace` or `FieldAppend` as appropriate.
5. Accept `SetControl` and `InvokeAction` as the only semantic inputs; keep GUI state out of the backend.
6. Update the backend package `README.md`.
7. Update `AGENTS.md` package map and extension points if the public surface changes.
8. Regenerate reference indexes: `python scripts/generate_indexes.py`.
9. When validating backend import/build performance, use representative small fixtures first and put explicit time bounds on large-asset probes. Do not benchmark giant morphology files by default.
