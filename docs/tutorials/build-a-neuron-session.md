---
title: Build a NEURON Session
summary: Minimal pattern for creating a live NEURON-backed application with the new session API.
---

# Build a NEURON Session

1. Subclass `NeuronSession`.
2. Implement `build_sections(...)`.
3. Implement `setup_model(...)` to insert mechanisms, clamps, synapses, or layout steps.
4. Wrap the session with `build_neuron_app(...)`.
5. Launch with `run_app(...)`.
6. On Windows, `run_app(...)` ignores spawned child imports internally, so the same shared launch code works across Windows, Linux, and macOS. Use `if __name__ == "__main__":` only if your script has extra top-level side effects you do not want repeated during `spawn`.

See `examples/neuron/visualizer_example.py` and `examples/neuron/multicell_example.py`.
