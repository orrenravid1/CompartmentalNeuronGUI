---
title: Architecture
summary: Implementation internals for contributors and advanced users — layer structure, protocol, frontend, and process docs.
---

# Architecture

These pages describe how CompNeuroVis is built, not how to use it. Read them when you are contributing, debugging a layer boundary issue, or need to understand how the system is wired before making a structural change.

- [Core Model](core-model.md) — how `Field`, `Geometry`, `Scene`, `Session`, and `Frontend` fit together. Start here.
- [Session Protocol](session-protocol.md) — session lifecycle, typed commands and updates, `BufferedSession` pattern, and `PipeTransport` behavior.
- [VisPy Frontend](vispy-frontend.md) — panel structure, refresh planning, state binding, interaction hooks, and how to extend the frontend.
- [Release Process](release-process.md) — version bumps, changelog, sealed branch tips, tags, and GitHub Releases.
- [PR Readiness Attestation](pr-readiness-attestation.md) — the attestation-only final commit workflow that binds readiness to a specific parent commit.
- [Design](design/index.md) — roadmap, settled decisions, backlog, and detailed proposals.
