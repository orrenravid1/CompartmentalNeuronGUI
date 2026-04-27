---
title: Build a Jaxley Session
summary: Step-by-step guide to building a live Jaxley-backed visualization by subclassing JaxleySession.
---

# Build a Jaxley Session

This tutorial shows the minimal pattern for a live Jaxley-backed simulation.
Use it when you want CompNeuroVis to drive a Jaxley model directly instead of
displaying static or replayed data.

Before you start, install the Jaxley extra:

```bash
pip install -e ".[jaxley]"
```

See `examples/jaxley/multicell_example.py` and the [Example Index](../reference/example-index.md) for fuller runnable examples.

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

## 3. Sample Custom Channel States Per Step

To record gating variables or other channel states alongside voltage, override
two hooks instead of `advance()`:

```python
import numpy as np

class MyJaxleySession(JaxleySession):
    def _sample_step(self):
        # Called once per simulation step.
        # Use _read_state() to read any channel variable at the display compartments.
        # State keys follow Jaxley's convention: 'ChannelName_statename'.
        return {
            "v": self._read_display_values(),
            "m": self._read_state("HH_m"),   # sodium activation gate
            "n": self._read_state("HH_n"),   # potassium gate
        }

    def _emit_batch(self, times_array, steps):
        # Called once per display update with all per-step results collected.
        latest = steps[-1]
        self.emit(FieldReplace(field_id="segment_display", values=latest["v"]))
        m_batch = np.stack([s["m"] for s in steps], axis=1)
        self.emit(FieldAppend(
            field_id="gating_history",
            append_dim="time",
            values=m_batch,
            coord_values=times_array,
            max_length=self.max_samples,
        ))
```

All channel states are available in `self._state` after each step - no additional
recording calls in `setup_model()` are needed. `_read_state()` reads any state
variable at the same compartment indices used for the morphology display.

## 4. Use the Default Geometry and Scene Builders (reference)

`JaxleySession` handles:

- building the live `jx.Network`
- recording a default displayed quantity
- converting Jaxley compartment geometry into `MorphologyGeometry`
- constructing the default `Scene`
- emitting live field updates

The geometry conversion step is handled by `JaxleySceneBuilder`. It turns Jaxley node and `xyzr` data into the frontend-facing morphology model.

That means user code usually does not need to construct geometry manually. The current sampled quantity is still voltage by default, but the default field roles are generic display/history roles rather than voltage-named fields.

## 5. Add Controls or Actions (optional)

```python
from compneurovis import ControlPresentationSpec, ControlSpec, ScalarValueSpec


def control_specs(self):
    return {
        "stim_amp": ControlSpec(
            id="stim_amp",
            label="Stimulus amplitude",
            value_spec=ScalarValueSpec(default=0.6, min=0.0, max=2.0, value_type="float"),
            presentation=ControlPresentationSpec(kind="slider", steps=100),
            send_to_session=True,
        ),
    }


def apply_control(self, control_id, value):
    if control_id == "stim_amp":
        self.stim_amp = float(value)
        self._rebuild_stimulus()
        return True
    return False
```

For a fuller runtime-update pattern, see `examples/jaxley/multicell_example.py` in the [Example Index](../reference/example-index.md), which rebuilds stimulus externals and refreshes runtime parameters after control changes.

Keep the controls semantic. The frontend will route `SetControl` and `InvokeAction` to the session for you.

## 6. Build and Run

```python
from compneurovis import build_jaxley_app, run_app

app = build_jaxley_app(MyJaxleySession, title="My Jaxley app")
run_app(app)
```

Pass the session class, not an already-created session instance. That keeps construction lazy and consistent with worker-backed transport behavior.

## 7. Customize the Default Scene (optional)

If you want to keep the default morphology/trace setup but tweak it, override `build_scene()`:

```python
def build_scene(self, *, geometry, display_values, time_value):
    scene = super().build_scene(
        geometry=geometry,
        display_values=display_values,
        time_value=time_value,
    )
    scene.replace_view("trace", {"rolling_window": 40.0})
    return scene
```

This is usually the right layer for adjusting the default views without redoing the whole backend integration.

## Loading and Positioning Cells

Jaxley-specific SWC, cache, and geometry helpers live under
`compneurovis.backends.jaxley.utils`.

```python
from compneurovis.backends.jaxley.utils import load_swc_jaxley, translate_cells_xyzr
```

Use that backend-owned package for Jaxley helpers.

Next steps:

- Read [Build a NEURON session](build-a-neuron-session.md) if you want the same live-session pattern with NEURON instead of Jaxley.
- Read [Session Update Model](../concepts/session-update-model.md) if you want the live backend/update contract behind `initialize()`, `advance()`, and emitted updates.
