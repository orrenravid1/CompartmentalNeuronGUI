from __future__ import annotations

from compneurovis.backends.neuron import NeuronSession
from compneurovis.core import AppSpec


def build_neuron_app(session: NeuronSession, *, title: str | None = None) -> AppSpec:
    return AppSpec(document=None, session=session, title=title or session.title)

