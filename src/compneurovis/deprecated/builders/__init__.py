from compneurovis.deprecated.builders.replay import ReplayBackend, build_replay_app
from compneurovis.deprecated.builders.surface import build_surface_app, grid_field

__all__ = ["ReplayBackend", "build_replay_app", "build_surface_app", "grid_field"]

try:  # optional backend dependency
    from compneurovis.deprecated.builders.neuron import build_neuron_app
except Exception:  # pragma: no cover - optional import
    build_neuron_app = None
else:
    __all__.append("build_neuron_app")

try:  # optional backend dependency
    from compneurovis.deprecated.builders.jaxley import build_jaxley_app
except Exception:  # pragma: no cover - optional import
    build_jaxley_app = None
else:
    __all__.append("build_jaxley_app")
