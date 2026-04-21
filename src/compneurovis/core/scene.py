from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from compneurovis.core.controls import ActionSpec, ControlSpec
from compneurovis.core.field import Field
from compneurovis.core.geometry import Geometry
from compneurovis.core.operators import OperatorSpec
from compneurovis.core.views import LinePlotViewSpec, MarkovGraphViewSpec, MorphologyViewSpec, SurfaceViewSpec, ViewSpec


PANEL_KIND_VIEW_3D = "view_3d"
PANEL_KIND_LINE_PLOT = "line_plot"
PANEL_KIND_CONTROLS = "controls"
PANEL_KIND_MARKOV_GRAPH = "markov_graph"


@dataclass(slots=True)
class PanelSpec:
    id: str
    kind: str
    view_ids: tuple[str, ...] = ()
    control_ids: tuple[str, ...] = ()
    action_ids: tuple[str, ...] = ()
    operator_ids: tuple[str, ...] = ()
    host_kind: str = "independent_canvas"
    title: str | None = None
    camera_distance: float | None = 200.0
    camera_elevation: float = 30.0
    camera_azimuth: float = 30.0


@dataclass(slots=True)
class LayoutSpec:
    title: str = "CompNeuroVis"
    panels: tuple[PanelSpec, ...] = ()
    panel_grid: tuple[tuple[str, ...], ...] = ()

    def resolved_panels(self) -> tuple[PanelSpec, ...]:
        return self.panels

    def panels_of_kind(self, kind: str) -> tuple[PanelSpec, ...]:
        return tuple(panel for panel in self.panels if panel.kind == kind)

    def panel(self, panel_id: str) -> PanelSpec | None:
        for panel in self.panels:
            if panel.id == panel_id:
                return panel
        return None

    def panel_for_view(self, view_id: str, *, kind: str | None = None) -> PanelSpec | None:
        for panel in self.panels:
            if kind is not None and panel.kind != kind:
                continue
            if view_id in panel.view_ids:
                return panel
        return None

    def patch_panel(self, panel_id: str, **changes) -> bool:
        """Apply ``dataclasses.replace`` changes to one panel in the panels tuple.

        Returns ``True`` if the panel was found and patched, ``False`` if not found.
        """
        for i, panel in enumerate(self.panels):
            if panel.id == panel_id:
                self.panels = (
                    *self.panels[:i],
                    replace(panel, **changes),
                    *self.panels[i + 1 :],
                )
                return True
        return False

    def replace_panels(
        self,
        panels: "tuple[PanelSpec, ...]",
        panel_grid: "tuple[tuple[str, ...], ...]" = (),
    ) -> None:
        """Replace the full panel inventory and optional grid. Does not re-normalize."""
        self.panels = panels
        self.panel_grid = panel_grid

    def normalize_panels(
        self,
        *,
        views: dict[str, ViewSpec],
        controls: dict[str, ControlSpec],
        actions: dict[str, ActionSpec],
    ) -> None:
        if self.panels:
            self.panels = self._normalize_explicit_panels(
                views=views,
                controls=controls,
                actions=actions,
            )
        else:
            self.panels = self._derive_default_panels(
                views=views,
                controls=controls,
                actions=actions,
            )

    def _normalize_explicit_panels(
        self,
        *,
        views: dict[str, ViewSpec],
        controls: dict[str, ControlSpec],
        actions: dict[str, ActionSpec],
    ) -> tuple[PanelSpec, ...]:
        normalized: list[PanelSpec] = []
        seen_panel_ids: set[str] = set()
        seen_view_ids: set[str] = set()

        for panel in self.panels:
            panel_id = panel.id.strip()
            if not panel_id:
                raise ValueError("PanelSpec.id must be non-empty")
            if panel_id in seen_panel_ids:
                raise ValueError(f"Duplicate panel id '{panel_id}'")

            if panel.kind == PANEL_KIND_VIEW_3D:
                view_ids = tuple(
                    dict.fromkeys(
                        view_id
                        for view_id in panel.view_ids
                        if view_id in views and isinstance(views[view_id], (MorphologyViewSpec, SurfaceViewSpec))
                    )
                )
                if not view_ids:
                    continue
                duplicate_view_ids = [view_id for view_id in view_ids if view_id in seen_view_ids]
                if duplicate_view_ids:
                    joined = ", ".join(duplicate_view_ids)
                    raise ValueError(f"3D view ids assigned to multiple panels: {joined}")
                seen_view_ids.update(view_ids)
                normalized.append(
                    PanelSpec(
                        id=panel_id,
                        kind=panel.kind,
                        view_ids=view_ids,
                        operator_ids=tuple(dict.fromkeys(operator_id for operator_id in panel.operator_ids if operator_id)),
                        host_kind=panel.host_kind,
                        title=panel.title,
                        camera_distance=panel.camera_distance,
                        camera_elevation=panel.camera_elevation,
                        camera_azimuth=panel.camera_azimuth,
                    )
                )
            elif panel.kind == PANEL_KIND_LINE_PLOT:
                view_ids = tuple(
                    dict.fromkeys(
                        view_id
                        for view_id in panel.view_ids
                        if view_id in views and isinstance(views[view_id], LinePlotViewSpec)
                    )
                )
                if not view_ids:
                    continue
                if len(view_ids) != 1:
                    raise ValueError(
                        f"Line plot panel '{panel_id}' must reference exactly one line-plot view id"
                    )
                duplicate_view_ids = [view_id for view_id in view_ids if view_id in seen_view_ids]
                if duplicate_view_ids:
                    joined = ", ".join(duplicate_view_ids)
                    raise ValueError(f"Line plot view ids assigned to multiple panels: {joined}")
                seen_view_ids.update(view_ids)
                normalized.append(
                    PanelSpec(
                        id=panel_id,
                        kind=panel.kind,
                        view_ids=view_ids,
                        title=panel.title,
                    )
                )
            elif panel.kind == PANEL_KIND_CONTROLS:
                control_ids = tuple(dict.fromkeys(control_id for control_id in panel.control_ids if control_id))
                action_ids = tuple(dict.fromkeys(action_id for action_id in panel.action_ids if action_id))
                if not control_ids and not action_ids:
                    control_ids = tuple(controls.keys())
                    action_ids = tuple(actions.keys())
                normalized.append(
                    PanelSpec(
                        id=panel_id,
                        kind=panel.kind,
                        control_ids=control_ids,
                        action_ids=action_ids,
                        title=panel.title,
                    )
                )
            elif panel.kind == PANEL_KIND_MARKOV_GRAPH:
                view_ids = tuple(
                    dict.fromkeys(
                        view_id
                        for view_id in panel.view_ids
                        if view_id in views and isinstance(views[view_id], MarkovGraphViewSpec)
                    )
                )
                if not view_ids:
                    continue
                if len(view_ids) != 1:
                    raise ValueError(
                        f"Markov graph panel '{panel_id}' must reference exactly one markov-graph view id"
                    )
                duplicate_view_ids = [view_id for view_id in view_ids if view_id in seen_view_ids]
                if duplicate_view_ids:
                    raise ValueError(
                        f"Markov graph view ids assigned to multiple panels: {', '.join(duplicate_view_ids)}"
                    )
                seen_view_ids.update(view_ids)
                normalized.append(
                    PanelSpec(
                        id=panel_id,
                        kind=panel.kind,
                        view_ids=view_ids,
                        title=panel.title,
                    )
                )
            else:
                raise ValueError(f"Unsupported panel kind '{panel.kind}'")

            seen_panel_ids.add(panel_id)

        return tuple(normalized)

    def _derive_default_panels(
        self,
        *,
        views: dict[str, ViewSpec],
        controls: dict[str, ControlSpec],
        actions: dict[str, ActionSpec],
    ) -> tuple[PanelSpec, ...]:
        panels: list[PanelSpec] = []

        for view in views.values():
            if isinstance(view, (MorphologyViewSpec, SurfaceViewSpec)):
                panels.append(
                    PanelSpec(
                        id=f"{view.id}-panel",
                        kind=PANEL_KIND_VIEW_3D,
                        view_ids=(view.id,),
                    )
                )
            elif isinstance(view, LinePlotViewSpec):
                panels.append(
                    PanelSpec(
                        id=f"{view.id}-panel",
                        kind=PANEL_KIND_LINE_PLOT,
                        view_ids=(view.id,),
                    )
                )
            elif isinstance(view, MarkovGraphViewSpec):
                panels.append(
                    PanelSpec(
                        id=f"{view.id}-panel",
                        kind=PANEL_KIND_MARKOV_GRAPH,
                        view_ids=(view.id,),
                    )
                )

        if controls or actions:
            panels.append(
                PanelSpec(
                    id="controls-panel",
                    kind=PANEL_KIND_CONTROLS,
                    control_ids=tuple(controls.keys()),
                    action_ids=tuple(actions.keys()),
                )
            )

        return tuple(panels)


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
        self.layout.normalize_panels(
            views=self.views,
            controls=self.controls,
            actions=self.actions,
        )

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
    diagnostics: "DiagnosticsSpec | None" = None


@dataclass(slots=True)
class DiagnosticsSpec:
    perf_log_enabled: bool = False
    perf_log_dir: str | Path | None = None
    perf_echo_stderr: bool = False
