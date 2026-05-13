from __future__ import annotations

from compneurovis.core import RunSpec
from compneurovis.backends import BackendBase, BackendSource


def build_neuron_app(
    backend: BackendSource,
    *,
    title: str | None = None,
    interaction_target=None,
) -> RunSpec:
    """Build a live app backed by a NeuronBackend subclass or backend factory."""

    if isinstance(backend, BackendBase):
        raise TypeError(
            "build_neuron_app() requires a Backend subclass or top-level zero-argument factory. "
            "Do not pass an already-created backend instance."
        )
    inferred_title = title
    return RunSpec(app_spec=None, backend=backend, interaction_target=interaction_target, title=inferred_title)
