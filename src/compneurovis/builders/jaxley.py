from __future__ import annotations

from compneurovis.core import RunSpec
from compneurovis.backends import BackendBase, BackendSource


def build_jaxley_app(
    backend: BackendSource,
    *,
    title: str | None = None,
    interaction_target=None,
) -> RunSpec:
    """Build a live app backed by a JaxleyBackend subclass or backend factory."""

    if isinstance(backend, BackendBase):
        raise TypeError(
            "build_jaxley_app() requires a Backend subclass or top-level zero-argument factory. "
            "Do not pass an already-created backend instance."
        )
    return RunSpec(app_spec=None, backend=backend, interaction_target=interaction_target, title=title)
