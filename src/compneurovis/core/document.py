from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from compneurovis.core.controls import ActionSpec, ControlSpec
from compneurovis.core.field import Field
from compneurovis.core.geometry import Geometry
from compneurovis.core.views import ViewSpec


@dataclass(slots=True)
class LayoutSpec:
    title: str = "CompNeuroVis"
    main_3d_view_id: str | None = None
    line_plot_view_id: str | None = None
    control_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class Document:
    fields: dict[str, Field]
    geometries: dict[str, Geometry]
    views: dict[str, ViewSpec]
    controls: dict[str, ControlSpec] = field(default_factory=dict)
    actions: dict[str, ActionSpec] = field(default_factory=dict)
    layout: LayoutSpec = field(default_factory=LayoutSpec)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.fields = dict(self.fields)
        self.geometries = dict(self.geometries)
        self.views = dict(self.views)
        self.controls = dict(self.controls)
        self.actions = dict(self.actions)
        self.metadata = dict(self.metadata)
        if not self.layout.control_ids:
            self.layout.control_ids = tuple(self.controls.keys())

    def replace_view(self, view_id: str, updates: dict[str, Any]) -> None:
        self.views[view_id] = replace(self.views[view_id], **updates)

    def replace_control(self, control_id: str, updates: dict[str, Any]) -> None:
        self.controls[control_id] = replace(self.controls[control_id], **updates)


@dataclass(slots=True)
class AppSpec:
    document: Document | None = None
    session: Any = None
    title: str | None = None

