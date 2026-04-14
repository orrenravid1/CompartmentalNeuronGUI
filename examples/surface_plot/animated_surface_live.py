"""
Animated surface — live computation approach. Renders the same radially-expanding sinc wave as
animated_surface_replay.py, but computes each frame on demand inside advance() rather than
pre-computing them all at startup. A speed control changes the wave propagation rate in real time.

Trade-offs vs the replay approach (see animated_surface_replay.py):
  - Negligible startup and memory cost (one frame computed at a time)
  - Small per-frame CPU cost (one numpy expression per advance() call)
  - Natural fit for parameter-driven or interactive computation
  - Controls can modify the computation itself, not just visual properties

Note: the current authoring pattern requires subclassing BufferedSession directly. A future
build_animated_surface_app(fn=...) builder is planned (see docs/architecture/design/backlog.md) that will make
this pattern available without writing a session class.

Run: python examples/surface_plot/animated_surface_live.py
"""

import numpy as np

from compneurovis import ActionSpec, ControlSpec, PanelSpec, SurfaceViewSpec, build_surface_app, grid_field, run_app
from compneurovis.core import AppSpec
from compneurovis.session import BufferedSession, SceneReady, FieldReplace, InvokeAction, Reset, SetControl

x = np.linspace(-4.0, 4.0, 120, dtype=np.float32)
y = np.linspace(-4.0, 4.0, 120, dtype=np.float32)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

field, geometry = grid_field(
    field_id="wave-height",
    values=np.zeros_like(R),
    x_coords=x,
    y_coords=y,
)

surface_view = SurfaceViewSpec(
    id="surface",
    title="animated sinc wave — live",
    field_id=field.id,
    geometry_id=geometry.id,
    color_map="bwr",
    color_limits=(-1.5, 2.0),
    render_axes=True,
    axes_in_middle=True,
    axis_labels=("x", "y", "height"),
    background_color="white",
    axis_color="black",
    text_color="black",
)

scene = build_surface_app(
    field=field,
    geometry=geometry,
    surface_view=surface_view,
    # send_to_session=True so the session receives SetControl when the slider moves.
    controls={"speed": ControlSpec("speed", "float", "Speed", 1.0, min=0.1, max=4.0, steps=78, send_to_session=True)},
    title="animated sinc wave — live",
    panels=(
        PanelSpec(id="surface-host", kind="view_3d", view_ids=("surface",), camera_distance=30.0),
        PanelSpec(id="controls-panel", kind="controls", control_ids=("speed",), action_ids=("pause", "reset")),
    ),
    panel_grid=(("surface-host",), ("controls-panel",)),
).scene

scene.actions["pause"] = ActionSpec("pause", "Pause / Resume", shortcuts=("Space",))
scene.actions["reset"] = ActionSpec("reset", "Reset", shortcuts=("R",))


class LiveAnimationSession(BufferedSession):
    def __init__(self):
        super().__init__()
        self._t = 0.0
        self._speed = 1.0
        self._playing = True

    def initialize(self):
        return scene

    def advance(self):
        if not self._playing:
            return
        self._t += 0.05 * self._speed
        Z = (np.sinc((R - self._t) / np.pi) * 2.0).astype(np.float32)
        # coords=None — the grid coordinates don't change, only the height values.
        self.emit(FieldReplace(field_id=field.id, values=Z, coords=None))

    def handle(self, command):
        if isinstance(command, SetControl) and command.control_id == "speed":
            self._speed = float(command.value)
        elif isinstance(command, InvokeAction) and command.action_id == "pause":
            self._playing = not self._playing
        elif isinstance(command, Reset):
            self._t = 0.0
            self._playing = True

    def idle_sleep(self):
        # ~30 fps update cadence
        return 1 / 30


run_app(AppSpec(session=LiveAnimationSession, title="animated sinc wave — live"))
