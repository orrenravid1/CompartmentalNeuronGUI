---
title: Build a NEURON Session
summary: Step-by-step guide to building a live NEURON-backed visualization by subclassing NeuronSession.
---

# Build a NEURON Session

This tutorial shows the minimal pattern for a live simulation. See `examples/neuron/visualizer_example.py` for a full working example.

## 1. Subclass NeuronSession

```python
from compneurovis import NeuronSession

class MyCellSession(NeuronSession):
    def build_sections(self):
        # Create and return your NEURON sections.
        # Called once before setup_model().
        soma = h.Section(name="soma")
        soma.L = 20; soma.diam = 20
        return [soma]

    def setup_model(self, sections):
        # Insert mechanisms, stimuli, and recorders.
        # sections is the list returned by build_sections().
        soma = sections[0]
        soma.insert("hh")
        self.iclamp = h.IClamp(soma(0.5))
        self.iclamp.amp = 0.5
        self.iclamp.dur = 500
```

`NeuronSession` handles:
- calling `build_sections()` and `setup_model()` at initialization
- building the `MorphologyGeometry` from your sections' 3-D coordinates
- constructing the `Document` and emitting `DocumentReady`
- stepping the simulation with `h.fadvance()` inside each `advance()` call
- emitting incremental voltage history with `FieldAppend` rather than resending full trace history every frame
- batching multiple internal simulation steps per frontend update via `display_dt`

You do not need to manage the NEURON run loop, pt3d parsing, or document construction yourself.

`NeuronSession(display_dt=...)` controls the simulation-time interval between frontend updates. The simulation still advances at `dt`; `display_dt` only affects how often batched live samples are emitted to the frontend.

## 2. Override Hooks (optional)

```python
def control_specs(self):
    # Return controls to show in the UI.
    return [ControlSpec("amp", "float", "Clamp amp (nA)", 0.5, min=0.0, max=2.0, steps=100)]

def action_specs(self):
    return [ActionSpec("reset", "Reset")]

def apply_control(self, control_id, value):
    if control_id == "amp":
        self.iclamp.amp = value

def apply_action(self, action_id, payload):
    if action_id == "reset":
        h.finitialize(-65)

def on_entity_clicked(self, entity_id, ctx):
    info = ctx.entity_info(entity_id)
    ctx.show_status(f"Clicked: {info['section_name']} @ {info['xloc']:.2f}")
    return True
```

## 3. Build and Run

```python
from compneurovis import build_neuron_app, run_app

session = MyCellSession()
app = build_neuron_app(session)
run_app(app)
```

On Windows, `run_app()` handles `spawn`-mode multiprocessing internally. You do not need `if __name__ == "__main__":` unless your script has top-level side effects you want to suppress in worker processes.

## Custom Layout

By default, `NeuronSession` shows a `MorphologyViewSpec` as the main 3-D view and a `LinePlotViewSpec` for the selected segment's voltage trace. To customize, override `build_document()` from `NeuronDocumentBuilder`:

```python
from compneurovis import NeuronDocumentBuilder, MorphologyViewSpec, LinePlotViewSpec

class MyCellSession(NeuronSession):
    ...
    def build_document_override(self, geometry, voltage_field):
        # Return a custom Document if you need non-default layout.
        ...
```

See `src/compneurovis/backends/neuron/document.py` for the default document construction logic.

## Loading from SWC

```python
from compneurovis.neuronutils.swc_utils import load_swc_neuron

class MyCellSession(NeuronSession):
    def build_sections(self):
        return load_swc_neuron("path/to/cell.swc")
```

`load_swc_neuron()` imports the SWC via NEURON's `Import3d` tool and returns a list of sections with 3-D coordinates already set.
