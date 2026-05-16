"""Sugar API for inline simulation visualization.

Matplotlib-style module-level API backed by a subprocess backend.
The simulation runs in a child process; Qt event loop stays in the main process.
No `if __name__ == '__main__':` guard required — role is determined via a
module-level flag set by _cnv_script_worker before the user script re-runs.
"""
from __future__ import annotations

import inspect
import multiprocessing as mp
import runpy
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from compneurovis.backends.base import BackendBase
from compneurovis.backends.host import BackendHost
from compneurovis.core.app import (
    AppSpec,
    DataCatalog,
    Field,
    InteractionCatalog,
    LayoutCatalog,
    LayoutSpec,
    LinePlotViewSpec,
    PanelSpec,
    ViewCatalog,
)
from compneurovis.core.controls import ActionSpec, ControlSpec, ScalarValueSpec
from compneurovis.core.hosts import configure_multiprocessing
from compneurovis.core.messages import (
    FieldAppend,
    FieldReplace,
    InvokeAction,
    Message,
    MessagePayload,
    SetControl,
    update_message,
)
from compneurovis.core.runtime import AppRuntime
from compneurovis.transports.pipe import PipeEndpoint, make_pipe_pair

# ---------------------------------------------------------------------------
# Backend slot — set by _cnv_script_worker before running user script.
# None  → main process (frontend mode)
# set   → subprocess (backend mode)
# ---------------------------------------------------------------------------

_backend_endpoint: PipeEndpoint | None = None


def _cnv_script_worker(script_path: str, endpoint: PipeEndpoint) -> None:
    """Entry point for the backend subprocess.

    Must be a top-level function in this module so multiprocessing can
    pickle it by qualified name (compneurovis.inline._cnv_script_worker).
    Do not move or rename without updating multiprocessing pickle target.
    """
    global _backend_endpoint, _app
    _backend_endpoint = endpoint
    # The bootstrap phase re-ran __main__ (spawn._fixup_main_from_path) and
    # polluted _app with duplicate bindings.  Reset before the real run.
    _app = InlineApp()
    runpy.run_path(script_path, run_name="__main__")


# ---------------------------------------------------------------------------
# Binding descriptors
# ---------------------------------------------------------------------------

SeriesReaders = Callable[[], float] | dict[str, Callable[[], float]]


@dataclass
class TraceBinding:
    name: str
    read: SeriesReaders
    x: Callable[[], float]
    rolling_window: float = 500.0
    y_min: float | None = None
    y_max: float | None = None
    y_unit: str = "a.u."
    x_unit: str = "ms"
    max_samples: int = 2400
    _field_id: str = field(init=False, default="")
    _view_id: str = field(init=False, default="")
    _buf_x: list = field(init=False, default_factory=list)
    _buf_vals: list = field(init=False, default_factory=list)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def _register(self, index: int) -> None:
        self._field_id = f"field_{index}_{self.name}"
        self._view_id = f"view_{index}_{self.name}"

    def _series(self) -> dict[str, Callable[[], float]]:
        if callable(self.read):
            return {self.name: self.read}
        return self.read

    def _sample(self) -> None:
        series = self._series()
        x = self.x()
        vals = [fn() for fn in series.values()]
        with self._lock:
            self._buf_x.append(x)
            self._buf_vals.append(vals)

    def _drain_message(self):
        with self._lock:
            if not self._buf_x:
                return None
            xs = self._buf_x[:]
            vals = self._buf_vals[:]
            self._buf_x.clear()
            self._buf_vals.clear()
        n_series = len(self._series())
        values = np.array(vals, dtype=np.float32).reshape(len(xs), n_series).T
        return update_message(FieldAppend(
            field_id=self._field_id,
            append_dim="time",
            values=values,
            coord_values=np.array(xs, dtype=np.float32),
            max_length=self.max_samples,
        ))

    def _initial_field(self) -> Field:
        series = self._series()
        return Field(
            id=self._field_id,
            values=np.array([[fn()] for fn in series.values()], dtype=np.float32),
            dims=("series", "time"),
            coords={
                "series": np.array(list(series.keys())),
                "time": np.array([self.x()], dtype=np.float32),
            },
            unit=self.y_unit,
        )

    def _view_spec(self) -> LinePlotViewSpec:
        series = self._series()
        return LinePlotViewSpec(
            id=self._view_id,
            title=self.name,
            field_id=self._field_id,
            x_dim="time",
            series_dim="series",
            x_unit=self.x_unit,
            y_unit=self.y_unit,
            rolling_window=self.rolling_window,
            trim_to_rolling_window=True,
            y_min=self.y_min,
            y_max=self.y_max,
            show_legend=len(series) > 1,
        )

    def _panel_spec(self) -> PanelSpec:
        return PanelSpec(id=f"panel_{self._view_id}", kind="line_plot", view_ids=(self._view_id,))

    def _replace_message(self):
        series = self._series()
        values = np.array([[fn()] for fn in series.values()], dtype=np.float32)
        return update_message(FieldReplace(
            field_id=self._field_id,
            values=values,
            coords={
                "series": np.array(list(series.keys())),
                "time": np.array([self.x()], dtype=np.float32),
            },
        ))


@dataclass
class ControlBinding:
    name: str
    label: str
    get: Callable[[], float]
    set: Callable[[Any], None]
    min: float = 0.0
    max: float = 1.0
    _control_id: str = field(init=False, default="")

    def _register(self, index: int) -> None:
        self._control_id = f"ctrl_{index}_{self.name}"

    def _control_spec(self) -> ControlSpec:
        return ControlSpec(
            id=self._control_id,
            label=self.label,
            value_spec=ScalarValueSpec(default=self.get(), min=self.min, max=self.max),
            send_to_backend=True,
        )


@dataclass
class ActionBinding:
    name: str
    label: str
    fn: Callable[[], None]
    resets_fields: bool = False
    _action_id: str = field(init=False, default="")

    def _register(self, index: int) -> None:
        self._action_id = f"action_{index}_{self.name}"

    def _action_spec(self) -> ActionSpec:
        return ActionSpec(id=self._action_id, label=self.label)


# ---------------------------------------------------------------------------
# InlineBackend — BackendBase wrapping user step function and bindings
# ---------------------------------------------------------------------------

class InlineBackend(BackendBase):
    """Backend actor for inline sugar sessions.

    Wraps the user's step function and trace/control/action bindings.
    Driven by BackendHost: handle() dispatches inbound commands;
    advance() runs simulation steps and emits FieldAppend updates.
    """

    _FRAME_MS = 1000.0 / 60.0

    def __init__(
        self,
        *,
        traces: list[TraceBinding],
        controls: list[ControlBinding],
        actions: list[ActionBinding],
        step: Callable[[], None] | None,
        dt_ms: float | None,
        speed: int | Callable[[], int] | None,
    ) -> None:
        super().__init__()
        self._traces = traces
        self._controls = controls
        self._actions = actions
        self._step_fn = step
        self._dt_ms = dt_ms
        self._speed = speed

    def handle(self, message: Message[MessagePayload]) -> None:
        payload = message.payload
        if isinstance(payload, SetControl):
            for c in self._controls:
                if c._control_id == payload.control_id:
                    c.set(payload.value)
                    break
        elif isinstance(payload, InvokeAction):
            for a in self._actions:
                if a._action_id == payload.action_id:
                    a.fn()
                    if a.resets_fields:
                        for t in self._traces:
                            self.emit_update(t._replace_message().payload)
                    break

    def advance(self) -> None:
        n = (
            self._speed() if callable(self._speed) else
            int(self._speed) if self._speed is not None else
            max(1, int(self._FRAME_MS / self._dt_ms)) if self._dt_ms else 1
        )
        if self._step_fn is not None:
            for _ in range(n):
                self._step_fn()
                for t in self._traces:
                    t._sample()
        for t in self._traces:
            msg = t._drain_message()
            if msg is not None:
                self.emit_update(msg.payload)

    def idle_sleep(self) -> float:
        return self._FRAME_MS / 1000.0


# ---------------------------------------------------------------------------
# InlineApp — accumulates bindings, builds AppSpec, orchestrates the run
# ---------------------------------------------------------------------------

class InlineApp:
    def __init__(self) -> None:
        self._title = "CompNeuroVis"
        self._traces: list[TraceBinding] = []
        self._controls: list[ControlBinding] = []
        self._actions: list[ActionBinding] = []

    def _add_trace(self, b: TraceBinding) -> None:
        b._register(len(self._traces))
        self._traces.append(b)

    def _add_control(self, b: ControlBinding) -> None:
        b._register(len(self._controls))
        self._controls.append(b)

    def _add_action(self, b: ActionBinding) -> None:
        b._register(len(self._actions))
        self._actions.append(b)

    def _build_app_spec(self) -> AppSpec:
        trace_panels = [t._panel_spec() for t in self._traces]
        ctrl_panel = PanelSpec(
            id="panel_controls",
            kind="controls",
            control_ids=tuple(c._control_id for c in self._controls),
            action_ids=tuple(a._action_id for a in self._actions),
        ) if self._controls or self._actions else None
        panels = tuple(trace_panels) + ((ctrl_panel,) if ctrl_panel else ())

        n = len(trace_panels)
        if ctrl_panel and n > 0:
            grid = tuple(
                (p.id, ctrl_panel.id) if i == 0 else (p.id,)
                for i, p in enumerate(trace_panels)
            )
        else:
            grid = None

        layout = LayoutSpec(title=self._title, panels=panels, panel_grid=grid)
        return AppSpec(
            data=DataCatalog(fields={t._field_id: t._initial_field() for t in self._traces}),
            view_catalog=ViewCatalog(views={t._view_id: t._view_spec() for t in self._traces}),
            interactions=InteractionCatalog(
                controls={c._control_id: c._control_spec() for c in self._controls},
                actions={a._action_id: a._action_spec() for a in self._actions},
            ),
            layout_catalog=LayoutCatalog.single(layout),
        )

    def show(
        self,
        *,
        step: Callable[[], None] | None = None,
        dt_ms: float | None = None,
        speed: int | Callable[[], int] | None = None,
        title: str = "CompNeuroVis",
    ) -> None:
        self._title = title
        if _backend_endpoint is not None:
            # Subprocess: run as backend actor via BackendHost.
            backend = InlineBackend(
                traces=self._traces,
                controls=self._controls,
                actions=self._actions,
                step=step,
                dt_ms=dt_ms,
                speed=speed,
            )
            host = BackendHost(endpoint=_backend_endpoint)
            host.start(lambda: backend, self._build_app_spec())
            try:
                while not host.should_stop():
                    started = time.monotonic()
                    host.step()
                    remaining = host.idle_sleep() - (time.monotonic() - started)
                    if remaining > 0:
                        time.sleep(remaining)
            except (BrokenPipeError, OSError):
                pass
            finally:
                host.stop()

        elif mp.current_process().name == "MainProcess":
            # Main process: orchestrate — spawn backend subprocess, run Qt frontend.
            # VispyFrontendWindow/Host imported here to avoid vispy Qt init in subprocess.
            from compneurovis.frontends.vispy.frontend import VispyFrontendWindow
            from compneurovis.frontends.vispy.host import VispyFrontendHost

            script_path = inspect.stack()[-1].filename
            app_spec = self._build_app_spec()
            pair = make_pipe_pair(left_name="frontend", right_name="backend")
            configure_multiprocessing()

            backend_process = mp.Process(
                target=_cnv_script_worker,
                args=(script_path, pair.right),
            )
            backend_process.start()
            pair.right.close()

            runtime = AppRuntime(app_spec=app_spec)
            frontend_host = VispyFrontendHost(
                actor_source=VispyFrontendWindow,
                runtime=runtime,
                endpoint=pair.left,
            )
            frontend_host.start()
            frontend_host.run()
            frontend_host.stop()

            backend_process.join(timeout=2)
            if backend_process.is_alive():
                backend_process.terminate()
                backend_process.join()
        # else: subprocess bootstrap re-import (spawn fixup_main) — do nothing.


# ---------------------------------------------------------------------------
# Module-level API
# ---------------------------------------------------------------------------

_app = InlineApp()


def trace(name: str, *, read: SeriesReaders, x: Callable[[], float], **kwargs) -> None:
    _app._add_trace(TraceBinding(name=name, read=read, x=x, **kwargs))


def control(
    name: str,
    *,
    label: str,
    get: Callable[[], float],
    set: Callable[[Any], None],
    min: float = 0.0,
    max: float = 1.0,
) -> None:
    _app._add_control(ControlBinding(name=name, label=label, get=get, set=set, min=min, max=max))


def action(
    name: str,
    *,
    label: str,
    fn: Callable[[], None],
    resets_fields: bool = False,
) -> None:
    _app._add_action(ActionBinding(name=name, label=label, fn=fn, resets_fields=resets_fields))


def show(
    step: Callable[[], None] | None = None,
    dt_ms: float | None = None,
    speed: int | Callable[[], int] | None = None,
    title: str = "CompNeuroVis",
) -> None:
    _app.show(step=step, dt_ms=dt_ms, speed=speed, title=title)


__all__ = [
    "TraceBinding",
    "ControlBinding",
    "ActionBinding",
    "InlineBackend",
    "InlineApp",
    "trace",
    "control",
    "action",
    "show",
    "_cnv_script_worker",
]
