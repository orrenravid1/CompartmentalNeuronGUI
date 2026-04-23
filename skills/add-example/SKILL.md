---
name: add-example
description: Add a new runnable example to CompNeuroVis. Use when demonstrating a new workflow, builder pattern, or visualization type that is not already covered by an existing example.
metadata:
  kind: authoring
  surface: examples
  stage: implement
  trust: general
---

# Add an Example

Reference implementations: `examples/surface_plot/static_surface_visualizer.py` (static) and `examples/neuron/complex_cell_example.py` (live session).

1. Place the file under `examples/`:
   - `examples/neuron/` for NEURON-backed live examples
   - `examples/jaxley/` for Jaxley-backed live examples
   - `examples/surface_plot/` for static, slice, or replay surface examples

2. Structure the file in this order:
   - top-of-file comment (one line) explaining what the example demonstrates
   - imports from `compneurovis`
   - data or model setup
   - Scene / AppSpec assembly
   - `run_app(app)` at the bottom, unguarded - `run_app()` handles spawn-mode multiprocessing internally

3. Keep module-level code free of expensive NEURON initialization. Any NEURON setup that runs at import time will repeat in every spawned worker process on Windows. Put it inside `build_sections()` or `setup_model()`.

4. Verify syntax/import safety without importing runnable entrypoints directly:
   - Use `python -m compileall examples`
   - Do **not** smoke-test a runnable example by importing its module in-process when it ends with unguarded `run_app(app)`. Importing that module will launch the app and can block indefinitely.
   - If you need runtime verification beyond `compileall`, launch the example as a separate process with an explicit timeout or do a manual GUI smoke-test.

5. When the example defines controls or actions, verify the session command path matches the UI:
   - `send_to_session=True` controls require `handle(SetControl(...))` support
   - action buttons require `handle(InvokeAction(...))` or equivalent dispatch
   - if the session resets control values from the backend, make sure the frontend control state is updated too, not just the model state

6. If the example is a primary entry point for a new workflow or backend, update authored docs (getting-started, README, AGENTS, relevant tutorials) using `update-docs-and-indexes`.

7. Regenerate the example index: `python scripts/generate_indexes.py`

8. Validate: `python scripts/generate_indexes.py --check`
