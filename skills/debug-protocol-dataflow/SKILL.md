---
name: debug-protocol-dataflow
description: Debug dataflow, transport, and session/frontend integration issues in CompNeuroVis. Use when DocumentReady, FieldUpdate, or DocumentPatch messages are missing, malformed, stale, or not producing the expected frontend behavior.
---

# Debug Protocol Dataflow

Read `docs/architecture/session-protocol.md` first.

Debug in this order:

1. Confirm the `Document` or `Field` shape is valid.
2. Confirm the session emits the expected typed update.
3. Confirm `PipeTransport` receives and forwards the update.
4. Confirm the frontend mutates `Document` state or view state correctly.
5. Confirm the target panel resolves state bindings as expected.

Use `python -m compileall src examples` and focused tests before launching the full GUI.
