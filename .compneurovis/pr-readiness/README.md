---
title: PR Readiness Receipts
summary: Directory reserved for commit-keyed PR readiness attestations that are added as standalone final commits on review-ready branches.
---

# PR Readiness Receipts

`python scripts/pr_readiness.py seal` writes one JSON receipt per sealed parent commit into this directory.

Each sealed branch tip should end with a final commit that adds exactly one receipt file here and nothing else, whether that tip is on a feature branch or on `main`.
