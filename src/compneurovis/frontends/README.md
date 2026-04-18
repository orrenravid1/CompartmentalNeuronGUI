---
title: Frontends Package
summary: Frontend implementations that consume Scene and SessionUpdate models.
---

# Frontends Package

`compneurovis.frontends` hosts renderer-specific frontends. The current
implementation is `vispy`, and it consumes `AppSpec` as the frontend contract,
including any app-scoped `DiagnosticsSpec` settings.
