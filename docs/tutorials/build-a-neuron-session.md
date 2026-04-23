---
title: Build a NEURON Session
summary: Step-by-step guide to building a live NEURON-backed visualization by subclassing NeuronSession.
---

# Build a NEURON Session

This tutorial shows the minimal pattern for a live NEURON-backed simulation.
Use it when you want CompNeuroVis to drive a real simulator session rather than
display static or replayed data.

Before you start, install the NEURON extra:

```bash
pip install -e ".[neuron]"
```

See `examples/neuron/complex_cell_example.py` for a fuller morphology-first
example. If you want a plot-only NEURON app with no morphology panel and a live
`IClamp` amplitude slider, see `examples/neuron/hh_point_model_controls.py`.
If you want a NEURON-backed signaling toy model built from bundled custom
point-process `.mod` files and a plot-only `BufferedSession`, see
`examples/neuron/signaling_cascade_vis.py`.

This is a backend-specific tutorial, not the conceptual boundary of the library. A NEURON-backed app may expose morphology, traces, surfaces, controls, or any combination of those features. Using `build_neuron_app(...)` today means "use NEURON as the backend with current default wiring," not "build a morphology-first app."

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
- constructing the `Scene` and emitting `SceneReady`
- stepping the simulation with `h.fadvance()` inside each `advance()` call
- calling `_sample_step()` each step and `_emit_batch()` each batch to collect and emit data
- emitting incremental display/history updates with `FieldAppend` rather than resending full trace history every frame
- batching multiple internal simulation steps per frontend update via `display_dt`

You do not need to manage the NEURON run loop, pt3d parsing, or scene construction yourself. Do not override `advance()` — use the sampling hooks below instead.

`NeuronSession(display_dt=...)` controls the simulation-time interval between frontend updates. The simulation still advances at `dt`; `display_dt` only affects how often batched live samples are emitted to the frontend.

## 2. Override Hooks (optional)

```python
from compneurovis import ControlPresentationSpec, ControlSpec, ScalarValueSpec

def control_specs(self):
    # Return controls to show in the UI.
    return {
        "amp": ControlSpec(
            id="amp",
            label="Clamp amp (nA)",
            value_spec=ScalarValueSpec(default=0.5, min=0.0, max=2.0, value_type="float"),
            presentation=ControlPresentationSpec(kind="slider", steps=100),
            send_to_session=True,
        )
    }

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

These hooks stay on the session class even for worker-backed apps. The library routes semantic commands such as clicks, keys, and actions to the worker session, so user code does not need to care about pipe/process boundaries.

## 3. Sample Custom Quantities Per Step

To record gating variables, ion currents, or any other per-step value alongside
voltage, override two hooks instead of `advance()`:

```python
class MySession(NeuronSession):
    def _sample_step(self):
        # Called once after each h.fadvance(). Return whatever you need per step.
        return {
            "v": self._read_display_values(),           # voltage across all segments
            "n": np.array([self.soma(0.5).n_hh]),       # potassium gate at soma
        }

    def _emit_batch(self, times_array, steps):
        # Called once per display update.
        # times_array: np.ndarray of simulation times, one per step.
        # steps: list of whatever _sample_step() returned, one per step.
        latest = steps[-1]
        self.emit(FieldReplace(field_id="segment_display", values=latest["v"]))
        n_batch = np.stack([s["n"] for s in steps], axis=1)
        self.emit(FieldAppend(
            field_id="gating_history",
            append_dim="time",
            values=n_batch,
            coord_values=times_array,
            max_length=self.max_samples,
        ))
```

`_sample_step()` runs inside the `fadvance()` loop — keep it fast. `_emit_batch()` runs after the loop and is where you build and emit your `FieldAppend` updates.

See `examples/neuron/hh_section_inspector.py` for a full worked example
sampling voltage, sodium activation/inactivation, and potassium gating across
multiple sections with linked line plots. See
`examples/neuron/hh_point_model_controls.py` for the complementary
single-compartment, plot-only pattern with a continuously adjustable live
clamp.

## 4. Record Scalar Variables

`record()` and `record_many()` let you attach named NEURON `_ref_*` pointers for
automatic per-step sampling without writing `_sample_step()`. Register them in
`_prepare_recorders()` and handle results in `on_recorded_samples()`:

```python
class MySession(NeuronSession):
    def _prepare_recorders(self):
        super()._prepare_recorders()
        # Record the potassium gating variable n_hh at the soma midpoint.
        self.record("n_soma", self.soma(0.5)._ref_n_hh)

    def on_recorded_samples(self, times, values):
        # times: 1-D float32 array, one entry per fadvance step in the batch.
        # values: dict mapping each registered name to a 1-D float32 array.
        n_batch = values["n_soma"]
        self.emit(FieldAppend(
            field_id="n_history",
            append_dim="time",
            values=n_batch[None, :],
            coord_values=times,
            max_length=self.max_samples,
        ))
```

Use `record()` when you need one scalar per step from a single point. Use
`_sample_step()` + `_emit_batch()` when you need arrays across many segments per step.

## 5. Build and Run

```python
from compneurovis import build_neuron_app, run_app

app = build_neuron_app(MyCellSession)
run_app(app)
```

Passing the session class keeps construction lazy inside the worker process. That avoids duplicate top-level session construction on Windows `spawn` while preserving the same simple user-facing launch pattern. For worker-backed apps, this is now the intended path; do not pass an already-created session instance.

`build_neuron_app(...)` is a current convenience helper. The long-term public model should stay feature-composable: choose a backend, then add traces, morphology, surfaces, controls, and layout as needed.

## Custom Layout

By default, `NeuronSession` shows a `MorphologyViewSpec` as the main 3-D view and a `LinePlotViewSpec` for the selected segment's trace. The default live contract is split:

- `segment_display`: latest values for current morphology coloring
- `segment_history`: retained trace history

The current sampled quantity in `NeuronSession` is still voltage by default, but the field roles are no longer voltage-named. `history_capture_mode=HistoryCaptureMode.ON_DEMAND` is the default. It keeps current display values live while retaining trace history only for segments the user actually asks to inspect. Use `HistoryCaptureMode.FULL` when the app needs full all-entity history for retrospective selection or playback.

To customize layout or views, override `build_scene()` from `NeuronSceneBuilder`:

```python
class MyCellSession(NeuronSession):
    ...
    def build_scene(self, *, geometry, display_values, time_value):
        scene = super().build_scene(
            geometry=geometry,
            display_values=display_values,
            time_value=time_value,
        )
        scene.replace_view("trace", {"rolling_window": 50.0})
        return scene
```

See `src/compneurovis/backends/neuron/scene.py` for the default scene construction logic.

## Loading from SWC

```python
from compneurovis.neuronutils import load_swc_neuron

class MyCellSession(NeuronSession):
    def build_sections(self):
        return load_swc_neuron("path/to/cell.swc")
```

`load_swc_neuron()` imports the SWC via NEURON's `Import3d` tool and returns a list of sections with 3-D coordinates already set.

Next steps:

- Read [Build a Jaxley session](build-a-jaxley-session.md) if you want the same live-session pattern with Jaxley instead of NEURON.
- Read [Session Update Model](../concepts/session-update-model.md) if you want the live backend/update contract behind `initialize()`, `advance()`, and emitted updates.
