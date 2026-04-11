"""Public authoring surface for CompNeuroVis."""

from compneurovis.builders import ReplaySession, build_replay_app, build_surface_app, grid_field
from compneurovis.core import (
    ActionSpec,
    AttributeRef,
    AppSpec,
    ControlSpec,
    Scene,
    Field,
    Geometry,
    GridGeometry,
    LayoutSpec,
    LinePlotViewSpec,
    MorphologyGeometry,
    MorphologyViewSpec,
    SeriesSpec,
    StateBinding,
    SurfaceViewSpec,
    View3DHostSpec,
    ViewSpec,
)
from compneurovis.frontends import VispyFrontendWindow, run_app
from compneurovis.session import HistoryCaptureMode

__all__ = [
    "ActionSpec",
    "AttributeRef",
    "AppSpec",
    "ControlSpec",
    "Scene",
    "Field",
    "Geometry",
    "GridGeometry",
    "HistoryCaptureMode",
    "LayoutSpec",
    "LinePlotViewSpec",
    "MorphologyGeometry",
    "MorphologyViewSpec",
    "ReplaySession",
    "SeriesSpec",
    "StateBinding",
    "SurfaceViewSpec",
    "View3DHostSpec",
    "ViewSpec",
    "VispyFrontendWindow",
    "build_replay_app",
    "build_surface_app",
    "grid_field",
    "run_app",
]

try:  # optional backend dependency
    from compneurovis.backends.neuron import NeuronSceneBuilder, NeuronSession
    from compneurovis.builders.neuron import build_neuron_app
except Exception:  # pragma: no cover - optional import
    NeuronSceneBuilder = None
    NeuronSession = None
    build_neuron_app = None
else:
    __all__.extend(["NeuronSceneBuilder", "NeuronSession", "build_neuron_app"])

try:  # optional backend dependency
    from compneurovis.backends.jaxley import JaxleySceneBuilder, JaxleySession
    from compneurovis.builders.jaxley import build_jaxley_app
except Exception:  # pragma: no cover - optional import
    JaxleySceneBuilder = None
    JaxleySession = None
    build_jaxley_app = None
else:
    __all__.extend(["JaxleySceneBuilder", "JaxleySession", "build_jaxley_app"])
