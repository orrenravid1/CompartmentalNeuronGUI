from __future__ import annotations

import numpy as np

from compneurovis import Field, LayoutSpec, PanelSpec, Scene, SurfaceViewSpec, build_replay_app
from compneurovis.session import FieldReplace, Reset


def _scene() -> Scene:
    field = Field(
        id="surface",
        values=np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        dims=("y", "x"),
        coords={
            "y": np.array([0.0, 1.0], dtype=np.float32),
            "x": np.array([0.0, 1.0], dtype=np.float32),
        },
    )
    return Scene(
        fields={field.id: field},
        geometries={},
        views={"surface": SurfaceViewSpec(id="surface", field_id=field.id)},
        layout=LayoutSpec(
            title="Replay test",
            panels=(
                PanelSpec(id="surface-panel", kind="view_3d", view_ids=("surface",)),
                PanelSpec(id="controls-panel", kind="controls"),
            ),
            panel_grid=(("surface-panel",), ("controls-panel",)),
        ),
    )


def test_build_replay_app_exposes_reset_action():
    scene = _scene()
    app = build_replay_app(
        scene=scene,
        field_id="surface",
        frames=[(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), None)],
    )

    assert app.scene is scene
    assert "reset" in scene.actions
    assert scene.actions["reset"].shortcuts == ("Space",)
    controls_panel = scene.layout.panel("controls-panel")
    assert controls_panel is not None
    assert controls_panel.action_ids == ("reset",)


def test_replay_session_reset_emits_first_frame():
    first = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    second = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    app = build_replay_app(
        scene=_scene(),
        field_id="surface",
        frames=[(first, None), (second, None)],
    )
    session = app.session()

    session.initialize()
    session.advance()
    session.read_updates()
    session.advance()
    session.read_updates()

    session.handle(Reset())
    updates = session.read_updates()

    assert len(updates) == 1
    assert isinstance(updates[0], FieldReplace)
    assert np.allclose(updates[0].values, first)
