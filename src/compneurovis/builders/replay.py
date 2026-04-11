from __future__ import annotations

from functools import partial

from compneurovis.core import ActionSpec, AppSpec, Scene
from compneurovis.session import BufferedSession, FieldReplace, Reset


class ReplaySession(BufferedSession):
    """Session that replays a precomputed sequence of frame replacements."""

    def __init__(self, *, scene: Scene, field_id: str, frames, interval_live: bool = True):
        super().__init__()
        self.scene = scene
        self.field_id = field_id
        self.frames = list(frames)
        self.index = 0
        self.interval_live = interval_live

    def initialize(self):
        return self.scene

    def is_live(self) -> bool:
        return self.interval_live

    def advance(self) -> None:
        if not self.frames:
            return
        values, coords = self.frames[self.index]
        self.emit(FieldReplace(field_id=self.field_id, values=values, coords=coords))
        self.index = (self.index + 1) % len(self.frames)

    def handle(self, command) -> None:
        if isinstance(command, Reset):
            self.index = 0
            if not self.frames:
                return None
            values, coords = self.frames[0]
            self.emit(FieldReplace(field_id=self.field_id, values=values, coords=coords))
        return None


def build_replay_app(*, scene: Scene, field_id: str, frames) -> AppSpec:
    """Build an app that replays precomputed frames through ReplaySession."""

    scene.actions.setdefault("reset", ActionSpec("reset", "Reset", shortcuts=("Space",)))
    if "reset" not in scene.layout.action_ids:
        scene.layout.action_ids = (*scene.layout.action_ids, "reset")

    return AppSpec(
        scene=scene,
        session=partial(ReplaySession, scene=scene, field_id=field_id, frames=frames),
        title=scene.layout.title,
    )
