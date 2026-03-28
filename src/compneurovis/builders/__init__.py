from compneurovis.builders.replay import ReplaySession, build_replay_app
from compneurovis.builders.surface import build_surface_app, grid_field

__all__ = ["ReplaySession", "build_replay_app", "build_surface_app", "grid_field"]

try:  # optional backend dependency
    from compneurovis.builders.neuron import build_neuron_app
except Exception:  # pragma: no cover - optional import
    build_neuron_app = None
else:
    __all__.append("build_neuron_app")
