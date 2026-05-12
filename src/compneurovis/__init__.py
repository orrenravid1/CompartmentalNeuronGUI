"""Public authoring surface for CompNeuroVis."""

from __future__ import annotations

from importlib import import_module

from compneurovis.backends import Backend, BackendSource, BufferedBackend, HistoryCaptureMode
from compneurovis.builders import ReplayBackend, build_replay_app, build_surface_app, grid_field
from compneurovis.core import (
    ActionSpec,
    AttributeRef,
    AppSpec,
    BoolValueSpec,
    ChoiceValueSpec,
    ControlPresentationSpec,
    ControlSpec,
    DiagnosticsSpec,
    Field,
    Geometry,
    GridGeometry,
    GridSliceOperatorSpec,
    LayoutSpec,
    LinePlotViewSpec,
    StateGraphViewSpec,
    MorphologyGeometry,
    MorphologyViewSpec,
    OperatorSpec,
    PanelSpec,
    RunSpec,
    ScalarValueSpec,
    SeriesSpec,
    StateBinding,
    SurfaceViewSpec,
    ViewSpec,
    XYValueSpec,
)
from compneurovis.frontends import Frontend, VispyFrontendWindow, run_app
from compneurovis.transports import PipeTransport, Transport

__all__ = [
    "ActionSpec",
    "AttributeRef",
    "AppSpec",
    "Backend",
    "BackendSource",
    "BoolValueSpec",
    "BufferedBackend",
    "ChoiceValueSpec",
    "ControlPresentationSpec",
    "ControlSpec",
    "DiagnosticsSpec",
    "Field",
    "Frontend",
    "Geometry",
    "GridGeometry",
    "GridSliceOperatorSpec",
    "HistoryCaptureMode",
    "LayoutSpec",
    "LinePlotViewSpec",
    "StateGraphViewSpec",
    "MorphologyGeometry",
    "MorphologyViewSpec",
    "OperatorSpec",
    "PanelSpec",
    "PipeTransport",
    "ReplayBackend",
    "RunSpec",
    "ScalarValueSpec",
    "SeriesSpec",
    "StateBinding",
    "SurfaceViewSpec",
    "Transport",
    "ViewSpec",
    "XYValueSpec",
    "VispyFrontendWindow",
    "build_replay_app",
    "build_surface_app",
    "grid_field",
    "run_app",
    "NeuronAppSpecBuilder",
    "NeuronBackend",
    "build_neuron_app",
    "JaxleyAppSpecBuilder",
    "JaxleyBackend",
    "build_jaxley_app",
]

_OPTIONAL_EXPORTS = {
    "NeuronAppSpecBuilder": ("compneurovis.backends.neuron", "NeuronAppSpecBuilder", "neuron"),
    "NeuronBackend": ("compneurovis.backends.neuron", "NeuronBackend", "neuron"),
    "build_neuron_app": ("compneurovis.builders.neuron", "build_neuron_app", "neuron"),
    "JaxleyAppSpecBuilder": ("compneurovis.backends.jaxley", "JaxleyAppSpecBuilder", "jaxley"),
    "JaxleyBackend": ("compneurovis.backends.jaxley", "JaxleyBackend", "jaxley"),
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
