---
title: NEURON Utils
summary: Helpers for importing, laying out, and serializing NEURON morphologies for CompNeuroVis backends and examples.
---

# NEURON Utils

`compneurovis.neuronutils` contains small NEURON-specific helpers used by the backend, builders, and morphology-loading examples.

Import them from the package entrypoint:

```python
from compneurovis.neuronutils import generate_layout, load_swc_neuron
```

Current helpers:

- `load_swc_neuron(path)`: import an SWC file directly into NEURON and return the instantiated sections
- `parse_swc(path)`: parse raw SWC nodes and child relationships for custom reconstruction
- `load_swc_multi(path, basename=None)`: rebuild every disconnected SWC tree as separately named NEURON section groups
- `generate_layout(sections, ...)`: synthesize simple `pt3d` geometry for sections that lack explicit morphology points
- `define_shape_layout()`: fallback to NEURON's built-in `h.define_shape()` layout
- `export_section_json(filename, sections)`: serialize section geometry, topology, biophysics, ions, and point processes
- `import_section_json(filename)`: rebuild sections and point processes from the exported JSON format
