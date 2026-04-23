"""
Signaling cascade viewer - NEURON point-process example using bundled custom mod files and a linked live plot.

Requires: NEURON and compiled bundled mechanisms under examples/neuron/signaling_cascade_mod/
Compile from inside that directory (for example `nrnivmodl.bat .` on Windows or `nrnivmodl .` on Unix/macOS).
Run: python examples/neuron/signaling_cascade_vis.py
"""

from __future__ import annotations

from collections import deque
import math
from pathlib import Path
import time

import numpy as np
from neuron import h, load_mechanisms

from compneurovis import (
    AppSpec,
    AttributeRef,
    ControlPresentationSpec,
    ControlSpec,
    Field,
    LayoutSpec,
    LinePlotViewSpec,
    PanelSpec,
    ScalarValueSpec,
    Scene,
    SeriesSpec,
    run_app,
)
from compneurovis.session import BufferedSession, FieldAppend, FieldReplace, Reset, SetControl


h.load_file("stdrun.hoc")


TITLE = "Signaling cascade viewer"
FIELD_ID = "cascade"
VIEW_ID = "cascade_plot"
MECHANISM_DIR = Path(__file__).with_name("signaling_cascade_mod")
DEFAULT_DT = 0.01
DEFAULT_DISPLAY_DT = 1.0


SERIES = (
    SeriesSpec("ligand_C", "Ligand C (uM)", source=AttributeRef("ligand", "C"), color=(100, 200, 255)),
    SeriesSpec("receptor_bound1", "Receptor bound1", source=AttributeRef("receptor", "bound1"), color=(255, 100, 100)),
    SeriesSpec("receptor_occupancy", "Receptor occupancy", source=AttributeRef("receptor", "occupancy"), color=(100, 255, 100)),
    SeriesSpec("receptor_activation", "Receptor activation", source=AttributeRef("receptor", "activation"), color=(200, 100, 255)),
    SeriesSpec("effector_s", "Effector s", source=AttributeRef("effector", "s"), color=(255, 165, 0)),
    SeriesSpec("effector_s_inf", "Effector s_inf", source=AttributeRef("effector", "s_inf"), color=(0, 0, 0)),
)


CONTROLS = (
    ControlSpec(
        id="external_input",
        label="Ligand external_input (uM/ms)",
        value_spec=ScalarValueSpec(default=0.005, min=0.0, max=0.1, value_type="float"),
        presentation=ControlPresentationSpec(kind="slider", steps=200),
        send_to_session=True,
        target=AttributeRef("ligand", "external_input"),
    ),
    ControlSpec(
        id="decay_rate",
        label="Ligand decay_rate (/ms)",
        value_spec=ScalarValueSpec(default=0.00955, min=1e-05, max=0.1, value_type="float"),
        presentation=ControlPresentationSpec(kind="slider", steps=200, scale="log"),
        send_to_session=True,
        target=AttributeRef("ligand", "decay_rate"),
    ),
    ControlSpec(
        id="kd1",
        label="Receptor Kd (uM)",
        value_spec=ScalarValueSpec(default=3.09, min=0.01, max=10.0, value_type="float"),
        presentation=ControlPresentationSpec(kind="slider", steps=200, scale="log"),
        send_to_session=True,
        target=AttributeRef("receptor", "kd1"),
    ),
    ControlSpec(
        id="efficacy1",
        label="Receptor efficacy1",
        value_spec=ScalarValueSpec(default=1.0, min=0.0, max=2.0, value_type="float"),
        presentation=ControlPresentationSpec(kind="slider", steps=200),
        send_to_session=True,
        target=AttributeRef("receptor", "efficacy1"),
    ),
    ControlSpec(
        id="decay1",
        label="Receptor decay1 (/ms)",
        value_spec=ScalarValueSpec(default=0.275, min=0.0001, max=1.0, value_type="float"),
        presentation=ControlPresentationSpec(kind="slider", steps=200, scale="log"),
        send_to_session=True,
        target=AttributeRef("receptor", "decay1"),
    ),
    ControlSpec(
        id="capacity",
        label="Receptor capacity",
        value_spec=ScalarValueSpec(default=1.62, min=0.0, max=5.0, value_type="float"),
        presentation=ControlPresentationSpec(kind="slider", steps=200),
        send_to_session=True,
        target=AttributeRef("receptor", "capacity"),
    ),
    ControlSpec(
        id="baseline_activity",
        label="Receptor baseline_activity",
        value_spec=ScalarValueSpec(default=0.0, min=0.0, max=1.0, value_type="float"),
        presentation=ControlPresentationSpec(kind="slider", steps=200),
        send_to_session=True,
        target=AttributeRef("receptor", "baseline_activity"),
    ),
    ControlSpec(
        id="s_min",
        label="Effector s_min",
        value_spec=ScalarValueSpec(default=0.0, min=0.0, max=1.0, value_type="float"),
        presentation=ControlPresentationSpec(kind="slider", steps=100),
        send_to_session=True,
        target=AttributeRef("effector", "s_min"),
    ),
    ControlSpec(
        id="s_max",
        label="Effector s_max",
        value_spec=ScalarValueSpec(default=1.0, min=0.0, max=2.0, value_type="float"),
        presentation=ControlPresentationSpec(kind="slider", steps=100),
        send_to_session=True,
        target=AttributeRef("effector", "s_max"),
    ),
    ControlSpec(
        id="K",
        label="Effector K (Hill midpoint)",
        value_spec=ScalarValueSpec(default=0.5, min=0.001, max=10.0, value_type="float"),
        presentation=ControlPresentationSpec(kind="slider", steps=200, scale="log"),
        send_to_session=True,
        target=AttributeRef("effector", "K"),
    ),
    ControlSpec(
        id="n",
        label="Effector hill coefficient n",
        value_spec=ScalarValueSpec(default=3, min=1, max=6, value_type="int"),
        send_to_session=True,
        target=AttributeRef("effector", "n"),
    ),
    ControlSpec(
        id="tau_on",
        label="Effector tau_on (ms)",
        value_spec=ScalarValueSpec(default=151.0, min=1.0, max=200.0, value_type="float"),
        presentation=ControlPresentationSpec(kind="slider", steps=200),
        send_to_session=True,
        target=AttributeRef("effector", "tau_on"),
    ),
    ControlSpec(
        id="tau_off",
        label="Effector tau_off (ms)",
        value_spec=ScalarValueSpec(default=140.0, min=1.0, max=200.0, value_type="float"),
        presentation=ControlPresentationSpec(kind="slider", steps=200),
        send_to_session=True,
        target=AttributeRef("effector", "tau_off"),
    ),
)


CONTROL_BY_ID = {control.id: control for control in CONTROLS}


def control_specs(*, display_dt: float = DEFAULT_DISPLAY_DT, dt: float = DEFAULT_DT) -> dict[str, ControlSpec]:
    controls: dict[str, ControlSpec] = {
        "display_dt": ControlSpec(
            id="display_dt",
            label="Simulation speed (ms sim/update)",
            value_spec=ScalarValueSpec(default=max(float(dt), float(display_dt)), min=float(dt), max=20.0, value_type="float"),
            presentation=ControlPresentationSpec(kind="slider", steps=199, scale="log"),
            send_to_session=True,
        )
    }
    controls.update({control.id: control for control in CONTROLS})
    return controls


def zero_series_history() -> dict[str, list[float]]:
    return {series.key: [0.0] for series in SERIES}


def ensure_bundled_mechanisms_loaded() -> None:
    if load_mechanisms(str(MECHANISM_DIR), warn_if_already_loaded=False):
        return
    raise RuntimeError(
        "Bundled NEURON mechanisms are not compiled. "
        f"From inside '{MECHANISM_DIR}', compile the .mod files so NEURON can load "
        "nrnmech.dll on Windows or the platform-specific libnrnmech.* directory on Unix/macOS."
    )


def build_cascade_scene(
    *,
    time_history: list[float],
    series_history: dict[str, list[float]],
    controls: dict[str, ControlSpec],
) -> Scene:
    field = Field(
        id=FIELD_ID,
        values=np.asarray([series_history[series.key] for series in SERIES], dtype=np.float32),
        dims=("series", "time"),
        coords={
            "series": np.asarray([series.label for series in SERIES]),
            "time": np.asarray(time_history, dtype=np.float32),
        },
        unit="a.u.",
    )
    view = LinePlotViewSpec(
        id=VIEW_ID,
        title="Signaling cascade",
        field_id=FIELD_ID,
        x_dim="time",
        series_dim="series",
        x_label="Time",
        x_unit="ms",
        y_label="Signal",
        series_colors={series.label: series.color for series in SERIES},
        show_legend=True,
        rolling_window=20.0,
        trim_to_rolling_window=True,
        y_min=0.0,
        y_max=1.8,
        x_major_tick_spacing=5.0,
        x_minor_tick_spacing=1.0,
        max_refresh_hz=60,
    )
    return Scene(
        fields={FIELD_ID: field},
        geometries={},
        views={VIEW_ID: view},
        controls=controls,
        layout=LayoutSpec(
            title=TITLE,
            panels=(
                PanelSpec(id="cascade-panel", kind="line_plot", view_ids=(VIEW_ID,)),
                PanelSpec(id="controls-panel", kind="controls", control_ids=tuple(controls.keys())),
            ),
        ),
    )


class SignalingCascadeSession(BufferedSession):
    @classmethod
    def startup_scene(cls) -> Scene | None:
        return build_cascade_scene(
            time_history=[0.0],
            series_history=zero_series_history(),
            controls=control_specs(),
        )

    def __init__(
        self,
        *,
        dt: float = DEFAULT_DT,
        display_dt: float = DEFAULT_DISPLAY_DT,
        v_init: float = -65.0,
        max_samples: int = 5000,
    ):
        super().__init__()
        self.dt = float(dt)
        if self.dt <= 0:
            raise ValueError("SignalingCascadeSession dt must be positive")
        self.display_dt = float(display_dt)
        if self.display_dt <= 0:
            raise ValueError("SignalingCascadeSession display_dt must be positive")
        self.display_dt = max(self.dt, self.display_dt)
        self.v_init = float(v_init)
        self.max_samples = int(max_samples)
        if self.max_samples <= 0:
            raise ValueError("SignalingCascadeSession max_samples must be positive")
        self.parameters = {control.id: control.default_value() for control in CONTROLS}
        self._time_history: deque[float] = deque(maxlen=self.max_samples)
        self._series_history: dict[str, deque[float]] = {
            series.key: deque(maxlen=self.max_samples) for series in SERIES
        }
        self.soma = None
        self.ligand = None
        self.receptor = None
        self.effector = None

    def control_specs(self) -> dict[str, ControlSpec]:
        return control_specs(display_dt=self.display_dt, dt=self.dt)

    def initialize(self):
        ensure_bundled_mechanisms_loaded()
        self._build_model()
        h.dt = self.dt
        h.finitialize(self.v_init)
        self._reset_history()
        self._capture_sample()
        return build_cascade_scene(
            time_history=list(self._time_history),
            series_history={key: list(values) for key, values in self._series_history.items()},
            controls=self.control_specs(),
        )

    def idle_sleep(self) -> float:
        return 1.0 / 60.0

    def advance(self) -> None:
        step_count = self._steps_per_update()
        time_values = np.empty(step_count, dtype=np.float32)
        batch_values = np.empty((len(SERIES), step_count), dtype=np.float32)
        for step_index in range(step_count):
            h.fadvance()
            sample_time, sample_values = self._capture_sample()
            time_values[step_index] = sample_time
            batch_values[:, step_index] = sample_values
        self.emit(self._field_append(time_values, batch_values))

    def handle(self, command) -> None:
        if isinstance(command, Reset):
            h.finitialize(self.v_init)
            self._reset_history()
            self._capture_sample()
            self.emit(self._field_replace())
            return
        if isinstance(command, SetControl):
            self.apply_control(command.control_id, command.value)

    def shutdown(self) -> None:
        self.soma = None
        self.ligand = None
        self.receptor = None
        self.effector = None

    def _build_model(self) -> None:
        t0 = time.perf_counter()
        soma = h.Section(name="soma")
        soma.L = 20.0
        soma.diam = 20.0
        soma.nseg = 1
        soma.Ra = 150.0
        soma.cm = 1.0
        soma.insert("pas")
        for seg in soma:
            seg.pas.g = 1e-4
            seg.pas.e = -65.0

        self.soma = soma
        self.ligand = h.GenericLigand(soma(0.5))
        self.ligand.C_init = 0.0
        self.receptor = h.GenericReceptor(soma(0.5))
        self.receptor.n_ligands = 1
        h.setpointer(self.ligand._ref_C, "C_lig1", self.receptor)
        h.setpointer(self.ligand._ref_C_init, "C_lig2", self.receptor)
        h.setpointer(self.ligand._ref_C_init, "C_lig3", self.receptor)
        h.setpointer(self.ligand._ref_C_init, "C_lig4", self.receptor)
        self.effector = h.SetpointRelaxEffector(soma(0.5))
        self.effector.K = 0.5
        h.setpointer(self.receptor._ref_activation, "drive", self.effector)

        for control in CONTROLS:
            self._apply_parameter_to_model(control.id)

        print(f"Signaling cascade cell built in {time.perf_counter() - t0:.2f}s")

    def _apply_parameter_to_model(self, control_id: str) -> None:
        control = CONTROL_BY_ID[control_id]
        value = self.parameters[control_id]
        if isinstance(control.value_spec, ScalarValueSpec) and control.value_spec.value_type == "int":
            value = int(round(float(value)))
        else:
            value = float(value)
        control.target.write(self, value)

    def _field_replace(self) -> FieldReplace:
        return FieldReplace(
            field_id=FIELD_ID,
            values=np.asarray([list(self._series_history[series.key]) for series in SERIES], dtype=np.float32),
            coords={
                "series": np.asarray([series.label for series in SERIES]),
                "time": np.asarray(list(self._time_history), dtype=np.float32),
            },
        )

    def _field_append(self, time_values: np.ndarray, batch_values: np.ndarray) -> FieldAppend:
        return FieldAppend(
            field_id=FIELD_ID,
            append_dim="time",
            values=batch_values,
            coord_values=time_values,
            max_length=self.max_samples,
        )

    def _reset_history(self) -> None:
        self._time_history = deque(maxlen=self.max_samples)
        self._series_history = {series.key: deque(maxlen=self.max_samples) for series in SERIES}

    def _steps_per_update(self) -> int:
        return max(1, int(math.ceil(self.display_dt / self.dt)))

    def _capture_sample(self) -> tuple[float, np.ndarray]:
        time_value = float(h.t)
        sample_values = np.asarray(
            [float(series.source.read(self)) for series in SERIES],
            dtype=np.float32,
        )
        self._time_history.append(time_value)
        for index, series in enumerate(SERIES):
            self._series_history[series.key].append(float(sample_values[index]))
        return time_value, sample_values

    def apply_control(self, name: str, value) -> bool:
        if name == "display_dt":
            try:
                self.display_dt = max(self.dt, float(value))
                return True
            except Exception:
                return False
        control = CONTROL_BY_ID.get(name)
        if control is None:
            return False
        try:
            if isinstance(control.value_spec, ScalarValueSpec) and control.value_spec.value_type == "int":
                coerced = int(round(float(value)))
            else:
                coerced = float(value)
            self.parameters[name] = coerced
            if control.target is not None and getattr(self, control.target.owner, None) is not None:
                self._apply_parameter_to_model(name)
            return True
        except Exception:
            return False


app = AppSpec(session=SignalingCascadeSession, title=TITLE)
run_app(app)
