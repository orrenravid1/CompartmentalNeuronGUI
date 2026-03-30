from __future__ import annotations

from compneurovis.backends.jaxley import JaxleySession
from compneurovis.core import AppSpec


def build_jaxley_app(session: JaxleySession, *, title: str | None = None) -> AppSpec:
    return AppSpec(document=None, session=session, interaction_target=session, title=title or session.title)
