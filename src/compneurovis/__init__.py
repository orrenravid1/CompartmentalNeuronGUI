"""Public authoring surface for CompNeuroVis."""

from __future__ import annotations

from importlib import import_module

from compneurovis.builders import ReplaySession, build_replay_app, build_surface_app, grid_field
from compneurovis.core import (
    ActionSpec,
    AttributeRef,
    AppSpec,
    ControlSpec,
    Field,
    Geometry,
    GridGeometry,
    GridSliceOperatorSpec,
    LayoutSpec,
    LinePlotViewSpec,
    MorphologyGeometry,
    MorphologyViewSpec,
    OperatorSpec,
    PanelSpec,
    Scene,
    SeriesSpec,
    StateBinding,
    SurfaceViewSpec,
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
    "GridSliceOperatorSpec",
    "HistoryCaptureMode",
    "LayoutSpec",
    "LinePlotViewSpec",
    "MorphologyGeometry",
    "MorphologyViewSpec",
    "OperatorSpec",
    "PanelSpec",
    "ReplaySession",
    "SeriesSpec",
    "StateBinding",
    "SurfaceViewSpec",
    "ViewSpec",
    "VispyFrontendWindow",
    "build_replay_app",
    "build_surface_app",
    "grid_field",
    "run_app",
    "NeuronSceneBuilder",
    "NeuronSession",
    "build_neuron_app",
    "JaxleySceneBuilder",
    "JaxleySession",
    "build_jaxley_app",
]

_OPTIONAL_EXPORTS = {
    "NeuronSceneBuilder": ("compneurovis.backends.neuron", "NeuronSceneBuilder", "neuron"),
    "NeuronSession": ("compneurovis.backends.neuron", "NeuronSession", "neuron"),
    "build_neuron_app": ("compneurovis.builders.neuron", "build_neuron_app", "neuron"),
    "JaxleySceneBuilder": ("compneurovis.backends.jaxley", "JaxleySceneBuilder", "jaxley"),
    "JaxleySession": ("compneurovis.backends.jaxley", "JaxleySession", "jaxley"),
    "build_jaxley_app": ("compneurovis.builders.jaxley", "build_jaxley_app", "jaxley"),
}


def __getattr__(name: str):
    target = _OPTIONAL_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name, extra_name = target
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Optional CompNeuroVis export {name!r} requires extra {extra_name!r}. "
            f'Install it with `pip install -e ".[{extra_name}]"`.'
        ) from exc

    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
