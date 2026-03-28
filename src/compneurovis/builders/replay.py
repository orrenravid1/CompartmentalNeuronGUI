from __future__ import annotations

from compneurovis.core import AppSpec, Document
from compneurovis.session import BufferedSession, FieldReplace


class ReplaySession(BufferedSession):
    def __init__(self, *, document: Document, field_id: str, frames, interval_live: bool = True):
        super().__init__()
        self.document = document
        self.field_id = field_id
        self.frames = list(frames)
        self.index = 0
        self.interval_live = interval_live

    def initialize(self):
        return self.document

    def is_live(self) -> bool:
        return self.interval_live

    def advance(self) -> None:
        if not self.frames:
            return
        values, coords = self.frames[self.index]
        self.emit(FieldReplace(field_id=self.field_id, values=values, coords=coords))
        self.index = (self.index + 1) % len(self.frames)

    def handle(self, command) -> None:
        return None


def build_replay_app(*, document: Document, field_id: str, frames) -> AppSpec:
    return AppSpec(document=document, session=ReplaySession(document=document, field_id=field_id, frames=frames), title=document.layout.title)
