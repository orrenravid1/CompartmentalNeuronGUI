"""
Animated surface — replay approach. Renders a radially-expanding sinc wave by cycling through a
pre-computed list of frames. Each step the session emits a FieldReplace with the next frame's
values; the surface updates in place without rebuilding geometry or axes.

All frames are computed at startup. The animation then runs with zero per-frame CPU cost —
each step is just an array lookup and a FieldReplace emit.

Trade-offs vs the live approach (see animated_surface_live.py):
  - Higher startup and memory cost (all frames held in memory)
  - Zero per-frame computation cost at runtime
  - Natural fit for pre-recorded data or fixed-length animations
  - No straightforward path to parameter-driven or interactive computation

Note: the current authoring pattern requires subclassing BufferedSession directly. A future
build_animated_surface_app(fn=...) builder is planned (see docs/architecture/design/backlog.md) that will make
this pattern available without writing a session class.

Run: python examples/surface_plot/animated_surface_replay.py
"""

import numpy as np

from compneurovis import ActionSpec, PanelSpec, SurfaceViewSpec, build_surface_app, grid_field, run_app
from compneurovis.core import AppSpec
from compneurovis.session import BufferedSession, SceneReady, FieldReplace, InvokeAction, Reset

N_FRAMES = 60

x = np.linspace(-4.0, 4.0, 120, dtype=np.float32)
y = np.linspace(-4.0, 4.0, 120, dtype=np.float32)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# Pre-compute all frames at startup. coords=None because the grid never changes — only heights do.
frames = []
for i in range(N_FRAMES):
    t = i / N_FRAMES * 2 * np.pi
    Z = (np.sinc((R - t) / np.pi) * 2.0).astype(np.float32)
    frames.append((Z, None))

field, geometry = grid_field(
    field_id="wave-height",
    values=frames[0][0],
    x_coords=x,
    y_coords=y,
)

surface_view = SurfaceViewSpec(
    id="surface",
    title="animated sinc wave — replay",
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
    controls={},
    title="animated sinc wave — replay",
    panels=(
        PanelSpec(id="surface-host", kind="view_3d", view_ids=("surface",), camera_distance=30.0),
        PanelSpec(id="controls-panel", kind="controls", action_ids=("pause", "reset")),
    ),
    panel_grid=(("surface-host",), ("controls-panel",)),
).scene

# Add explicit playback controls so the example exposes buttons in the controls panel.
scene.actions["pause"] = ActionSpec("pause", "Pause / Resume", shortcuts=("Space",))
scene.actions["reset"] = ActionSpec("reset", "Reset", shortcuts=("R",))


class ReplayAnimationSession(BufferedSession):
    def __init__(self):
        super().__init__()
        self._index = 0
        self._playing = True

    def initialize(self):
        return scene

    def advance(self):
        if not self._playing:
            return
        values, coords = frames[self._index]
        self.emit(FieldReplace(field_id=field.id, values=values, coords=coords))
        self._index = (self._index + 1) % len(frames)

    def handle(self, command):
        if isinstance(command, InvokeAction) and command.action_id == "pause":
            self._playing = not self._playing
        elif isinstance(command, Reset):
            self._index = 0
            self._playing = True

    def idle_sleep(self):
        # ~30 fps playback cadence
        return 1 / 30


run_app(AppSpec(session=ReplayAnimationSession, title="animated sinc wave — replay"))
