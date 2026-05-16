from __future__ import annotations

from functools import partial

from compneurovis.core import ActionSpec, AppSpec, RunSpec
from compneurovis.backends import BackendBase
from compneurovis.core.messages import FieldReplace, Reset


class ReplayBackend(BackendBase):
    """Backend that replays a precomputed sequence of frame replacements."""

    def __init__(self, *, app_spec: AppSpec, field_id: str, frames, interval_live: bool = True):
        super().__init__()
        self.app_spec = app_spec
        self.field_id = field_id
        self.frames = list(frames)
        self.index = 0
        self.interval_live = interval_live

    def initialize(self, app_spec: AppSpec) -> None:
        pass

    def is_live(self) -> bool:
        return self.interval_live

    def update(self) -> None:
        if not self.frames:
            return
        values, coords = self.frames[self.index]
        self.emit_update(FieldReplace(field_id=self.field_id, values=values, coords=coords))
        self.index = (self.index + 1) % len(self.frames)

    def handle(self, message) -> None:
        command = message.payload
        if isinstance(command, Reset):
            self.index = 0
            if not self.frames:
                return None
            values, coords = self.frames[0]
            self.emit_update(FieldReplace(field_id=self.field_id, values=values, coords=coords))
        return None


def build_replay_app(*, app_spec: AppSpec, field_id: str, frames) -> RunSpec:
    """Build an app that replays precomputed frames through ReplayBackend."""

    app_spec.interactions.actions.setdefault("reset", ActionSpec("reset", "Reset", shortcuts=("Space",)))
    app_spec.active_layout().normalize_panels(
        views=app_spec.view_catalog.views,
        controls=app_spec.interactions.controls,
        actions=app_spec.interactions.actions,
    )

    return RunSpec(
        app_spec=app_spec,
        backend=partial(ReplayBackend, app_spec=app_spec, field_id=field_id, frames=frames),
        title=app_spec.active_layout().title,
    )
