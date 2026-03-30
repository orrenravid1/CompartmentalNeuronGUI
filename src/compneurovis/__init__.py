from compneurovis.builders import ReplaySession, build_replay_app, build_surface_app, grid_field
from compneurovis.core import (
    ActionSpec,
    AppSpec,
    ControlSpec,
    Document,
    Field,
    Geometry,
    GridGeometry,
    LayoutSpec,
    LinePlotViewSpec,
    MorphologyGeometry,
    MorphologyViewSpec,
    StateBinding,
    SurfaceViewSpec,
    ViewSpec,
)
from compneurovis.frontends import VispyFrontendWindow, run_app

__all__ = [
    "ActionSpec",
    "AppSpec",
    "ControlSpec",
    "Document",
    "Field",
    "Geometry",
    "GridGeometry",
    "LayoutSpec",
    "LinePlotViewSpec",
    "MorphologyGeometry",
    "MorphologyViewSpec",
    "ReplaySession",
    "StateBinding",
    "SurfaceViewSpec",
    "ViewSpec",
    "VispyFrontendWindow",
    "build_replay_app",
    "build_surface_app",
    "grid_field",
    "run_app",
]

try:  # optional backend dependency
    from compneurovis.backends.neuron import NeuronDocumentBuilder, NeuronSession
    from compneurovis.builders.neuron import build_neuron_app
except Exception:  # pragma: no cover - optional import
    NeuronDocumentBuilder = None
    NeuronSession = None
    build_neuron_app = None
else:
    __all__.extend(["NeuronDocumentBuilder", "NeuronSession", "build_neuron_app"])

try:  # optional backend dependency
    from compneurovis.backends.jaxley import JaxleyDocumentBuilder, JaxleySession
    from compneurovis.builders.jaxley import build_jaxley_app
except Exception:  # pragma: no cover - optional import
    JaxleyDocumentBuilder = None
    JaxleySession = None
    build_jaxley_app = None
else:
    __all__.extend(["JaxleyDocumentBuilder", "JaxleySession", "build_jaxley_app"])
