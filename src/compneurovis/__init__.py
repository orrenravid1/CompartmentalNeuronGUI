"""Public authoring surface for CompNeuroVis."""

from __future__ import annotations

from importlib import import_module

from compneurovis.backends import BackendBase, HistoryCaptureMode
from compneurovis.builders import ReplayBackend, build_replay_app, build_surface_app, grid_field
from compneurovis.core import (
    ActionSpec,
    ActorBase,
    ActorRole,
    ActorSpec,
    AttributeRef,
    AppSpec,
    BoolValueSpec,
    ChoiceValueSpec,
    ControlPresentationSpec,
    ControlSpec,
    DataCatalog,
    DiagnosticsSpec,
    Field,
    Geometry,
    GridGeometry,
    GridSliceOperatorSpec,
    InteractionCatalog,
    LayoutCatalog,
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
    ViewCatalog,
    XYValueSpec,
)
from compneurovis.frontends import FrontendBase, VispyFrontendHost, VispyFrontendWindow
from compneurovis.run import run_app
from compneurovis.messages import (
    CommandMessage,
    Message,
    MessagePayload,
    MessageType,
    UpdateMessage,
    command_message,
    make_message,
    message_type_for_payload,
    update_message,
)
from compneurovis.transports import PipeEndpoint, Transport, pipe_transport

__all__ = [
    "ActionSpec",
    "ActorBase",
    "ActorRole",
    "ActorSpec",
    "AttributeRef",
    "AppSpec",
    "BackendBase",
    "BoolValueSpec",
    "ChoiceValueSpec",
    "CommandMessage",
    "ControlPresentationSpec",
    "ControlSpec",
    "DataCatalog",
    "DiagnosticsSpec",
    "Field",
    "FrontendBase",
    "Geometry",
    "GridGeometry",
    "GridSliceOperatorSpec",
    "HistoryCaptureMode",
    "InteractionCatalog",
    "LayoutCatalog",
    "LayoutSpec",
    "LinePlotViewSpec",
    "Message",
    "MessagePayload",
    "MessageType",
    "MorphologyGeometry",
    "MorphologyViewSpec",
    "OperatorSpec",
    "PanelSpec",
    "PipeEndpoint",
    "ReplayBackend",
    "RunSpec",
    "ScalarValueSpec",
    "SeriesSpec",
    "StateBinding",
    "StateGraphViewSpec",
    "SurfaceViewSpec",
    "Transport",
    "UpdateMessage",
    "ViewCatalog",
    "ViewSpec",
    "VispyFrontendHost",
    "VispyFrontendWindow",
    "XYValueSpec",
    "build_replay_app",
    "build_surface_app",
    "command_message",
    "grid_field",
    "make_message",
    "message_type_for_payload",
    "pipe_transport",
    "run_app",
    "update_message",
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
