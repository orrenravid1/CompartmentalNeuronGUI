"""Universal notebook frontend — works in VS Code, JupyterLab, classic Jupyter.

Morphology panel: vispy offscreen → ipywidgets.Image (fast OpenGL coloring).
Trace panel:      ipympl matplotlib figure (interactive zoom/pan, live update).
Combined in an ipywidgets VBox.

Architecture:
    NotebookFrontend    — FrontendBase actor; owns rendering state and widget tree.
    NotebookFrontendHost— FrontendHost; owns the asyncio poll loop and AppRuntime ref.

Requires: ipympl, ipyevents  (pip install ipympl ipyevents)
"""
from __future__ import annotations

import asyncio
import io
import time
from typing import Any

import numpy as np

from compneurovis.core.app import ActorRole, ActorSpec, AppSpec, RunSpec
from compneurovis.core.geometry import MorphologyGeometry
from compneurovis.core.messages import FieldReplace, Message, StopBackend, command_message
from compneurovis.core.runtime import AppRuntime
from compneurovis.frontends.base import FrontendBase
from compneurovis.frontends.host import FrontendHost
from compneurovis.transports.pipe import PipeEndpoint

POLL_HZ = 30
MAX_SAMPLES = 4000
RENDER_HZ = 15


def _ensure_vispy_backend() -> None:
    from vispy.app import _default_app as _da
    if _da.default_app is None:
        from vispy import use
        use(app="pyqt6", gl="gl+")


# --------------------------------------------------------------------------- #
# Actor                                                                        #
# --------------------------------------------------------------------------- #

class NotebookFrontend(FrontendBase):
    """Morphology + trace notebook actor.

    Owns all rendering state. Stateless with respect to the transport — the
    host drives receive / flush / renders via the ActorHost contract.
    """

    def __init__(
        self,
        *,
        dt: float = 0.025,
        segment_index: int = 0,
        morph_size: tuple[int, int] = (800, 320),
        trace_figsize: tuple[float, float] = (8, 2.5),
        ylim: tuple[float, float] = (-90.0, 60.0),
        y_label: str = "V (mV)",
    ) -> None:
        super().__init__()
        self._dt = dt
        self._segment_index = segment_index
        self._display_field_id = "segment_display"
        self._voltages: np.ndarray | None = None
        self._buf: list[float] = []
        self._step = 0
        self._last_render = 0.0
        self._render_due = False
        self._morph_dirty = False
        self.stop_requested = False  # host checks this flag

        self._color_map = "scalar"
        self._color_limits: tuple[float, float] | None = (-80.0, 50.0)
        self._color_norm = "auto"

        # ------------------------------------------------------------------ #
        # Morphology panel — vispy offscreen canvas                           #
        # ------------------------------------------------------------------ #
        _ensure_vispy_backend()
        from vispy import scene
        from vispy.scene.cameras import TurntableCamera

        canvas = scene.SceneCanvas(
            keys="interactive", bgcolor="black", show=False, size=morph_size,
        )
        view = canvas.central_widget.add_view()
        view.camera = TurntableCamera(
            fov=60, distance=200, elevation=30, azimuth=30,
            translate_speed=100, up="+z",
        )
        self._morph_canvas = canvas
        from compneurovis.frontends.vispy.renderers.morphology import MorphologyRenderer
        self._morph_renderer = MorphologyRenderer(view)

        import ipywidgets as widgets
        self._morph_widget = widgets.Image(format="png", width=morph_size[0], height=morph_size[1])
        self._camera = view.camera

        # Mouse state for drag-to-rotate
        self._mouse_down = False
        self._mouse_last: tuple[int, int] = (0, 0)

        from ipyevents import Event
        morph_events = Event(
            source=self._morph_widget,
            watched_events=["mousedown", "mouseup", "mousemove", "wheel", "mouseleave", "dragstart"],
            prevent_default_action=True,
        )
        morph_events.on_dom_event(self._on_mouse_event)

        # ------------------------------------------------------------------ #
        # Trace panel — ipympl interactive matplotlib figure                  #
        # ------------------------------------------------------------------ #
        import matplotlib
        matplotlib.use("module://ipympl.backend_nbagg")
        import matplotlib.pyplot as plt

        self._plt = plt
        plt.ioff()
        fig, ax = plt.subplots(figsize=trace_figsize)
        fig.patch.set_facecolor("#111111")
        ax.set_facecolor("#111111")
        for spine in ax.spines.values():
            spine.set_color("#555555")
        ax.tick_params(colors="white")
        ax.set_xlabel("t (ms)", color="white")
        ax.set_ylabel(y_label, color="white")
        ax.set_ylim(*ylim)
        (self._trace_line,) = ax.plot([], [], color="#4fc3f7", lw=0.8)
        ax.set_xlim(0, 100)
        fig.tight_layout(pad=0.4)
        self._fig = fig
        self._ax = ax

        # Stop button; host wires up the actual stop() call after start()
        stop_btn = widgets.Button(
            description="Stop", button_style="danger",
            layout=widgets.Layout(width="80px"),
        )
        stop_btn.on_click(lambda _: setattr(self, "stop_requested", True))
        self._widget = widgets.VBox([self._morph_widget, fig.canvas, stop_btn])

    # ---------------------------------------------------------------------- #
    # ActorBase contract                                                       #
    # ---------------------------------------------------------------------- #

    def initialize(self, app_spec: AppSpec) -> None:
        for geo in app_spec.data.geometries.values():
            if isinstance(geo, MorphologyGeometry):
                self._morph_renderer.set_geometry(geo)
                n = len(geo.positions)
                self._voltages = np.full(n, -65.0, dtype=np.float32)
                break

        field = app_spec.data.fields.get(self._display_field_id)
        if field is not None and field.values is not None:
            vals = np.asarray(field.values, dtype=np.float32)
            if vals.ndim > 1:
                vals = vals[:, -1]
            self._voltages = vals
            if len(vals) > self._segment_index:
                self._buf.append(float(vals[self._segment_index]))

        from compneurovis.core.views import MorphologyViewSpec
        for view_spec in app_spec.view_catalog.views.values():
            if isinstance(view_spec, MorphologyViewSpec):
                self._color_map = view_spec.color_map or "scalar"
                self._color_norm = view_spec.color_norm or "auto"
                if view_spec.color_limits is not None and not isinstance(view_spec.color_limits, str):
                    self._color_limits = tuple(view_spec.color_limits)  # type: ignore[assignment]
                break

        self._render_morph()

    def handle(self, message: Message) -> None:
        payload = message.payload
        if not (isinstance(payload, FieldReplace) and payload.field_id == self._display_field_id):
            return
        vals = np.asarray(payload.values, dtype=np.float32)
        if vals.ndim > 1:
            vals = vals[:, -1]
        self._voltages = vals
        self._buf.append(float(vals[min(self._segment_index, len(vals) - 1)]))
        self._step += 1
        self._render_due = True

    # ---------------------------------------------------------------------- #
    # Rendering (called by host step loop)                                    #
    # ---------------------------------------------------------------------- #

    def flush_renders(self, now: float) -> None:
        """Render morph+trace if timing is due; render morph if camera dirty."""
        if self._render_due and now - self._last_render >= 1.0 / RENDER_HZ:
            _PERF_LOG = r"c:\Users\orren\Documents\PythonProjects\CompNeuroVis\scratch\perf_stats.txt"
            t0 = time.monotonic()
            self._render_morph()
            t1 = time.monotonic()
            self._render_trace()
            t2 = time.monotonic()
            with open(_PERF_LOG, "a") as _f:
                _f.write(f"[flush_renders] morph_ms={(t1-t0)*1000:.1f} trace_ms={(t2-t1)*1000:.1f} buf={len(self._buf)}\n")
            self._last_render = now
            self._render_due = False
        if self._morph_dirty:
            self._render_morph()
            self._morph_dirty = False

    # ---------------------------------------------------------------------- #

    def _on_mouse_event(self, event: dict) -> None:
        etype = event.get("type")
        x, y = event.get("offsetX", 0), event.get("offsetY", 0)

        if etype == "mousedown":
            self._mouse_down = True
            self._mouse_last = (x, y)
        elif etype in ("mouseup", "mouseleave"):
            self._mouse_down = False
        elif etype == "mousemove" and self._mouse_down:
            dx = x - self._mouse_last[0]
            dy = y - self._mouse_last[1]
            self._mouse_last = (x, y)
            self._camera.azimuth -= dx * 0.5
            self._camera.elevation = float(np.clip(self._camera.elevation + dy * 0.5, -90, 90))
            self._morph_dirty = True
        elif etype == "wheel":
            delta = event.get("deltaY", 0)
            self._camera.distance *= 1.0 + delta * 0.001
            self._morph_dirty = True

    def _render_morph(self) -> None:
        _PERF_LOG = r"c:\Users\orren\Documents\PythonProjects\CompNeuroVis\scratch\perf_stats.txt"
        import time as _t
        ta = _t.monotonic()
        n_v = len(self._voltages) if self._voltages is not None else -1
        if self._voltages is not None:
            self._morph_renderer.update_colors(
                self._voltages, self._color_map,
                color_limits=self._color_limits, color_norm=self._color_norm,
            )
        tb = _t.monotonic()
        rgba = self._morph_canvas.render()
        tc = _t.monotonic()
        buf = io.BytesIO()
        from PIL import Image
        Image.fromarray(rgba).save(buf, format="png")
        td = _t.monotonic()
        self._morph_widget.value = buf.getvalue()
        te = _t.monotonic()
        with open(_PERF_LOG, "a") as _f:
            _f.write(
                f"[render_morph] n_voltages={n_v} "
                f"update_colors_ms={(tb-ta)*1000:.1f} "
                f"canvas_render_ms={(tc-tb)*1000:.1f} "
                f"pil_ms={(td-tc)*1000:.1f} "
                f"widget_ms={(te-td)*1000:.1f} "
                f"rgba_shape={rgba.shape}\n"
            )

    def _render_trace(self) -> None:
        y = np.asarray(self._buf[-MAX_SAMPLES:], dtype=np.float32)
        n = len(y)
        if n < 2:
            return
        t_end = self._step * self._dt
        t_start = max(0.0, t_end - n * self._dt)
        x = np.linspace(t_start, t_end, n)
        self._trace_line.set_data(x, y)
        self._ax.set_xlim(max(0.0, t_end - MAX_SAMPLES * self._dt), max(t_end, 10.0))
        self._fig.canvas.draw_idle()


# --------------------------------------------------------------------------- #
# Host                                                                         #
# --------------------------------------------------------------------------- #

class NotebookFrontendHost(FrontendHost):
    """Drives the notebook frontend actor. Mirrors VispyFrontendHost for Qt.

    Owns the asyncio poll loop and holds an AppRuntime reference for
    coordinated startup/shutdown. The transport endpoint is injectable —
    in-process queue today, WebSocket tomorrow.
    """

    def __init__(
        self,
        runtime: AppRuntime,
        endpoint: PipeEndpoint,
        *,
        dt: float = 0.025,
        segment_index: int = 0,
        morph_size: tuple[int, int] = (800, 320),
        trace_figsize: tuple[float, float] = (8, 2.5),
        ylim: tuple[float, float] = (-90.0, 60.0),
        y_label: str = "V (mV)",
    ) -> None:
        super().__init__(endpoint=endpoint)
        self._runtime = runtime
        self._frontend_kwargs = dict(
            dt=dt,
            segment_index=segment_index,
            morph_size=morph_size,
            trace_figsize=trace_figsize,
            ylim=ylim,
            y_label=y_label,
        )
        self._running = False
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        actor_source = lambda: NotebookFrontend(**self._frontend_kwargs)
        super().start(actor_source, self._runtime.app_spec)
        self._running = True

    def run(self) -> Any:
        """Kick off asyncio poll loop and return the VBox widget."""
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._poll_loop())
        return self._notebook_frontend()._widget

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self.endpoint is not None:
            try:
                self.endpoint.send(command_message(StopBackend()))
            except Exception:
                pass
        if self._task is not None:
            self._task.cancel()
            self._task = None
        self._runtime.stop()
        super().stop()

    async def _poll_loop(self) -> None:
        import time as _t
        _PERF_LOG = r"c:\Users\orren\Documents\PythonProjects\CompNeuroVis\scratch\perf_stats.txt"
        with open(_PERF_LOG, "w") as _f:
            _f.write("=== poll_loop start (new/stashed version) ===\n")
        _poll_count = 0
        interval = 1.0 / POLL_HZ
        frontend = self._notebook_frontend()
        while self._running:
            if frontend.stop_requested:
                self.stop()
                break
            try:
                t0 = _t.monotonic()
                self.receive()
                frontend.flush_renders(time.monotonic())
                self.flush()
                _poll_count += 1
                if _poll_count % 30 == 0:
                    poll_ms = (_t.monotonic() - t0) * 1000
                    with open(_PERF_LOG, "a") as _f:
                        _f.write(f"t={_t.monotonic():.3f} [poll_tick] n={_poll_count} poll_ms={poll_ms:.1f}\n")
            except (BrokenPipeError, OSError):
                self._running = False
                break
            await asyncio.sleep(interval)

    def _notebook_frontend(self) -> NotebookFrontend:
        actor = self._actor()
        if not isinstance(actor, NotebookFrontend):
            raise TypeError(f"NotebookFrontendHost expected NotebookFrontend, got {type(actor)!r}")
        return actor


# --------------------------------------------------------------------------- #
# Launch helper                                                                #
# --------------------------------------------------------------------------- #

def _launch_notebook(
    *,
    backend_factory,
    app_spec: AppSpec,
    dt: float = 0.025,
) -> Any:
    """Start backend thread + notebook frontend and return the VBox widget.

    Compiles to a RunSpec and calls start_app() so the full architecture
    (AppRuntime, ActorSpec, transport) is exercised uniformly.

    Parameters
    ----------
    backend_factory : zero-arg callable returning a BackendBase instance
    app_spec        : AppSpec built from the backend before calling this
    dt              : simulation timestep in ms (for the trace time axis)
    """
    from compneurovis.backends.host import ThreadBackendHost
    from compneurovis.core.run import start_app
    from compneurovis.core.app import RoutingSpec
    from compneurovis.transports import routed_transport

    routing = RoutingSpec(
        control_routes={
            control_id: ("backend",)
            for control_id, control in app_spec.interactions.controls.items()
            if control.send_to_backend
        },
        action_routes={
            action_id: ("backend",)
            for action_id in app_spec.interactions.actions
        },
        default_command_targets=("backend",),
        default_update_targets=("frontend",),
    )

    handle = start_app(RunSpec(
        app_spec=app_spec,
        actors=[
            ActorSpec(
                id="backend",
                role=ActorRole.BACKEND,
                host_source=lambda r, ep, _f=backend_factory: ThreadBackendHost(_f, r, ep),
            ),
            ActorSpec(
                id="frontend",
                role=ActorRole.FRONTEND,
                host_source=lambda r, ep: NotebookFrontendHost(r, ep, dt=dt),
                runs_in_foreground=False,
            ),
        ],
        transport=routed_transport(routing, mode="inprocess"),
        routing=routing,
    ))
    return handle.widget("frontend")


__all__ = ["NotebookFrontend", "NotebookFrontendHost"]
