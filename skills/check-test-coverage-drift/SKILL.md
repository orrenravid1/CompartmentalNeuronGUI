---
name: check-test-coverage-drift
description: Audit whether a CompNeuroVis change set is still covered by the current tests and add or update tests when behavior, protocol contracts, builders, docs validations, or examples change. Use before merging code when test expectations may have drifted from the implementation.
---

# Check Test Coverage Drift

Run this workflow after implementation changes and before calling a change PR-ready.

1. Identify the touched code paths and the user-visible behavior or invariants they changed.
2. Map each change to the closest existing test area:
   - `tests/test_field.py`
   - `tests/test_geometry.py`
   - `tests/test_pipe_transport.py`
   - `tests/test_frontend_bindings.py`
   - `tests/test_docs_and_indexes.py`
3. Decide whether the current tests still cover:
   - validation rules
   - serialization/protocol behavior
   - frontend state bindings
   - docs/index invariants
   - example importability or builder behavior
4. Add or update tests when a change introduces new branches, new public contracts, or new failure modes.
5. Run the smallest meaningful verification first, then expand:
   - targeted `pytest` modules
   - optional backend suites such as `pytest --run-jaxley tests/test_jaxley_backend.py` when the change touches that backend
   - `pytest`
   - `python -m compileall src examples tests`

Use this decision rule:

- If behavior changed and no existing assertion would fail on a regression, tests are stale.
- If a public builder, protocol message, or documented workflow changed, add or update tests in the same change.
- If the change is limited to comments or non-behavioral text, call that out explicitly instead of adding noise tests.

Keep the output concrete: state what is covered, what is not, and what tests were added or should be added.
