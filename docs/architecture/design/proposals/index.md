---
title: Proposals
summary: Detailed design documents for specific subsystem changes not yet implemented.
---

# Proposals

Each proposal covers a specific feature or subsystem change in enough detail to inform implementation. Proposals are not settled decisions — when a proposal is implemented, the durable lessons move to [Design Decisions](../decisions.md).

- [Layout Workbench](layout-workbench-proposal.md) — replace the transitional panel grid with a recursive split-tree model and uniform panel identity.
- [Runtime Panel Layout Updates](panel-layout-updates.md) — `PanelPatch` and `LayoutReplace` for session-driven panel changes without full scene rebuild.
- [WebSocket Transport](websocket-transport-proposal.md) — add a WebSocket transport to support WSL backends, Windows frontends, and future non-Python clients.
