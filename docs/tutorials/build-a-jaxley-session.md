---
title: Build a Jaxley Session
summary: Step-by-step guide to building a live Jaxley-backed visualization by subclassing JaxleySession.
---

# Build a Jaxley Session

This tutorial shows the minimal pattern for a live Jaxley simulation. See [examples/jaxley/multicell_example.py](/c:/Users/orren/Documents/PythonProjects/CompartmentalNeuronGUI/examples/jaxley/multicell_example.py) for a fuller example.

## 1. Subclass JaxleySession

```python
import jaxley as jx
from jaxley.channels import HH

from compneurovis import JaxleySession


class MyJaxleySession(JaxleySession):
    def build_cells(self):
        comp = jx.Compartment()
        branch = jx.Branch([comp] * 5)
        cell = jx.Cell([branch], parents=[-1])
        cell.insert(HH())
        return [cell]
```

`build_cells()` returns either one `jx.Cell` or an iterable of cells.

## 2. Configure the Network

Override `setup_model()` when you need channels, synapses, recordings, or stimuli beyond the cell construction step.

```python
def setup_model(self, network, cells):
    del cells
    network.cell(0).branch(0).loc(0.5).stimulate(0.6)
```

If you need a custom network instead of the default `jx.Network(cells)`, override `build_network()`.

## 3. Use the Default Geometry and Document Builders

`JaxleySession` handles:

- building the live `jx.Network`
- recording a default displayed quantity
- converting Jaxley compartment geometry into `MorphologyGeometry`
- constructing the default `Document`
- emitting live field updates

The geometry conversion step is handled by `JaxleyDocumentBuilder`. It turns Jaxley node and `xyzr` data into the frontend-facing morphology model.

That means user code usually does not need to construct geometry manually. The current sampled quantity is still voltage by default, but the default field roles are generic display/history roles rather than voltage-named fields.

## 4. Add Controls or Actions (optional)

```python
from compneurovis import ControlSpec


def control_specs(self):
    return {
        "stim_amp": ControlSpec("stim_amp", "float", "Stimulus amplitude", 0.6, min=0.0, max=2.0, steps=100),
    }


def apply_control(self, control_id, value):
    if control_id == "stim_amp":
        self.stim_amp = float(value)
        self._rebuild_stimulus()
        return True
    return False
```

For a fuller runtime-update pattern, see [examples/jaxley/multicell_example.py](/c:/Users/orren/Documents/PythonProjects/CompartmentalNeuronGUI/examples/jaxley/multicell_example.py), which rebuilds stimulus externals and refreshes runtime parameters after control changes.

Keep the controls semantic. The frontend will route `SetControl` and `InvokeAction` to the session for you.

## 5. Build and Run

```python
from compneurovis import build_jaxley_app, run_app

app = build_jaxley_app(MyJaxleySession, title="My Jaxley app")
run_app(app)
```

Pass the session class, not an already-created session instance. That keeps construction lazy and consistent with worker-backed transport behavior.

## 6. Customize the Default Document (optional)

If you want to keep the default morphology/trace setup but tweak it, override `build_document()`:

```python
def build_document(self, *, geometry, display_values, time_value):
    document = super().build_document(
        geometry=geometry,
        display_values=display_values,
        time_value=time_value,
    )
    document.replace_view("trace", {"rolling_window": 40.0})
    return document
```

This is usually the right layer for adjusting the default views without redoing the whole backend integration.
