from __future__ import annotations

from compneurovis.core import AppSpec
from compneurovis.session import Session, SessionSource


def build_neuron_app(
    session: SessionSource,
    *,
    title: str | None = None,
    interaction_target=None,
) -> AppSpec:
    """Build a live app backed by a NeuronSession subclass or session factory."""

    if isinstance(session, Session):
        raise TypeError(
            "build_neuron_app() requires a Session subclass or top-level zero-argument factory. "
            "Do not pass an already-created session instance."
        )
    inferred_title = title
    return AppSpec(scene=None, session=session, interaction_target=interaction_target, title=inferred_title)
