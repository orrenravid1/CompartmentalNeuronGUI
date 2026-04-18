---
title: Session Package
summary: Session base classes, typed protocol, and local pipe transport.
---

# Session Package

`compneurovis.session` contains:

- `Session` and `BufferedSession`
- protocol commands and updates
- `PipeTransport`

`PipeTransport` mirrors app-scoped diagnostics into spawned worker processes so
frontend and session perf logging can be configured explicitly rather than only
through inherited shell environment.

