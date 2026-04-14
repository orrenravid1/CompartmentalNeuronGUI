---
name: add-simulator-backend
description: Add or update a simulator backend/session for CompNeuroVis. Use when creating a new live or replay backend under compneurovis.backends, wiring it to the Session protocol, or documenting how a backend should emit SceneReady, FieldReplace or FieldAppend, StatePatch, Status, and ScenePatch updates.
metadata:
  kind: authoring
  surface: backend
  stage: implement
  trust: general
---

# Add a Simulator Backend

Read `docs/architecture/core-model.md` and `docs/architecture/session-protocol.md` first.

Reference implementation: `src/compneurovis/backends/neuron/session.py`.

1. Create a package under `src/compneurovis/backends/<name>`.
2. Subclass `Session` or `BufferedSession`; emit only typed protocol updates - no direct frontend calls.
3. Emit `SceneReady` with a fully-built `Scene` on initialization.
4. Express all measured or simulated values as `Field` objects, delivered via `FieldReplace` or `FieldAppend` as appropriate. Emit `StatePatch`, `Status`, and `ScenePatch` only for semantic state, status text, or metadata/view/control changes.
5. Handle the semantic command set the app needs. Common commands include `Reset`, `SetControl`, `InvokeAction`, `EntityClicked`, and `KeyPressed`; keep raw GUI state out of the backend.
6. Update the backend package `README.md`.
7. Update `AGENTS.md` package map and extension points if the public surface changes.
8. Regenerate reference indexes: `python scripts/generate_indexes.py`.
9. When validating backend import/build performance, use representative small fixtures first and put explicit time bounds on large-asset probes. Do not benchmark giant morphology files by default.
