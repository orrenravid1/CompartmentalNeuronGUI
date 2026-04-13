from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from compneurovis.core.controls import ActionSpec, ControlSpec
from compneurovis.core.field import Field
from compneurovis.core.geometry import Geometry
from compneurovis.core.operators import OperatorSpec
from compneurovis.core.views import ViewSpec


@dataclass(slots=True)
class View3DHostSpec:
    id: str
    view_ids: tuple[str, ...]
    operator_ids: tuple[str, ...] = ()
    kind: str = "independent_canvas"
    title: str | None = None
    camera_distance: float | None = 200.0
    camera_elevation: float = 30.0
    camera_azimuth: float = 30.0

    def normalized(self) -> "View3DHostSpec | None":
        view_ids = tuple(dict.fromkeys(view_id for view_id in self.view_ids if view_id))
        operator_ids = tuple(dict.fromkeys(operator_id for operator_id in self.operator_ids if operator_id))
        if not view_ids:
            return None
        host_id = self.id or view_ids[0]
        return View3DHostSpec(
            id=host_id,
            view_ids=view_ids,
            operator_ids=operator_ids,
            kind=self.kind,
            title=self.title,
            camera_distance=self.camera_distance,
            camera_elevation=self.camera_elevation,
            camera_azimuth=self.camera_azimuth,
        )


@dataclass(slots=True)
class LayoutSpec:
    title: str = "CompNeuroVis"
    main_3d_view_id: str | None = None
    view_3d_ids: tuple[str, ...] = ()
    view_3d_hosts: tuple[View3DHostSpec, ...] = ()
    line_plot_view_ids: tuple[str, ...] = ()
    control_ids: tuple[str, ...] = ()
    action_ids: tuple[str, ...] = ()

    def resolved_3d_view_ids(self) -> tuple[str, ...]:
        if self.view_3d_ids:
            return tuple(dict.fromkeys(view_id for view_id in self.view_3d_ids if view_id))
        if self.main_3d_view_id is None:
            return ()
        return (self.main_3d_view_id,)

    def resolved_3d_hosts(self) -> tuple[View3DHostSpec, ...]:
        if self.view_3d_hosts:
            resolved_hosts: list[View3DHostSpec] = []
            seen_view_ids: set[str] = set()
            for host in self.view_3d_hosts:
                normalized = host.normalized()
                if normalized is None:
                    continue
                filtered_view_ids = tuple(view_id for view_id in normalized.view_ids if view_id not in seen_view_ids)
                if not filtered_view_ids:
                    continue
                seen_view_ids.update(filtered_view_ids)
                resolved_hosts.append(
                    View3DHostSpec(
                        id=normalized.id,
                        view_ids=filtered_view_ids,
                        operator_ids=normalized.operator_ids,
                        kind=normalized.kind,
                        title=normalized.title,
                        camera_distance=normalized.camera_distance,
                        camera_elevation=normalized.camera_elevation,
                        camera_azimuth=normalized.camera_azimuth,
                    )
                )
            return tuple(resolved_hosts)
        return tuple(
            View3DHostSpec(id=view_id, view_ids=(view_id,), kind="independent_canvas")
            for view_id in self.resolved_3d_view_ids()
        )

    def normalize_3d_views(self) -> None:
        resolved_hosts = self.resolved_3d_hosts()
        resolved_view_ids = tuple(
            dict.fromkeys(
                view_id
                for host in resolved_hosts
                for view_id in host.view_ids
            )
        )
        self.view_3d_hosts = resolved_hosts
        self.view_3d_ids = resolved_view_ids
        self.main_3d_view_id = resolved_view_ids[0] if resolved_view_ids else None

    def resolved_line_plot_view_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(view_id for view_id in self.line_plot_view_ids if view_id))

    def normalize_line_plots(self) -> None:
        resolved_view_ids = self.resolved_line_plot_view_ids()
        self.line_plot_view_ids = resolved_view_ids


@dataclass(slots=True)
class Scene:
    fields: dict[str, Field]
    geometries: dict[str, Geometry]
    views: dict[str, ViewSpec]
    operators: dict[str, OperatorSpec] = field(default_factory=dict)
    controls: dict[str, ControlSpec] = field(default_factory=dict)
    actions: dict[str, ActionSpec] = field(default_factory=dict)
    layout: LayoutSpec = field(default_factory=LayoutSpec)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.fields = dict(self.fields)
        self.geometries = dict(self.geometries)
        self.views = dict(self.views)
        self.operators = dict(self.operators)
        self.controls = dict(self.controls)
        self.actions = dict(self.actions)
        self.metadata = dict(self.metadata)
        self.layout.normalize_3d_views()
        self.layout.normalize_line_plots()
        if not self.layout.control_ids:
            self.layout.control_ids = tuple(self.controls.keys())
        if not self.layout.action_ids:
            self.layout.action_ids = tuple(self.actions.keys())

    def replace_view(self, view_id: str, updates: dict[str, Any]) -> None:
        self.views[view_id] = replace(self.views[view_id], **updates)

    def replace_operator(self, operator_id: str, updates: dict[str, Any]) -> None:
        self.operators[operator_id] = replace(self.operators[operator_id], **updates)

    def replace_control(self, control_id: str, updates: dict[str, Any]) -> None:
        self.controls[control_id] = replace(self.controls[control_id], **updates)


@dataclass(slots=True)
class AppSpec:
    scene: Scene | None = None
    session: Any = None
    interaction_target: Any = None
    title: str | None = None
