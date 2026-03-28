---
title: Session Protocol
summary: Session lifecycle and typed command/update protocol for local transports.
---

# Session Protocol

`Session` is the live/replay execution interface. The current protocol uses:

- Commands: `Reset`, `SetControl`, `InvokeAction`
- Updates: `DocumentReady`, `FieldUpdate`, `DocumentPatch`, `Status`, `Error`

The initial implementation only ships `PipeTransport`, but the message model is intentionally frontend-agnostic so a future websocket transport can reuse it.

