from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from compneurovis.core import (
    GridSliceOperatorSpec,
    LinePlotViewSpec,
    MorphologyViewSpec,
    Scene,
    StateBinding,
    StateGraphViewSpec,
    SurfaceViewSpec,
)
from compneurovis.core.scene import (
    PANEL_KIND_LINE_PLOT,
    PANEL_KIND_STATE_GRAPH,
    PANEL_KIND_VIEW_3D,
)


@dataclass(frozen=True, slots=True)
class RefreshTarget:
    kind: str
    view_id: str | None = None

    @classmethod
    def controls(cls) -> "RefreshTarget":
        return cls("controls")

    @classmethod
    def line_plot(cls, view_id: str) -> "RefreshTarget":
        return cls("line_plot", view_id)

    @classmethod
    def morphology(cls, view_id: str) -> "RefreshTarget":
        return cls("morphology", view_id)

    @classmethod
    def surface_visual(cls, view_id: str) -> "RefreshTarget":
        return cls("surface_visual", view_id)

    @classmethod
    def surface_style(cls, view_id: str) -> "RefreshTarget":
        return cls("surface_style", view_id)

    @classmethod
    def surface_axes_geometry(cls, view_id: str) -> "RefreshTarget":
        return cls("surface_axes_geometry", view_id)

    @classmethod
    def surface_axes_style(cls, view_id: str) -> "RefreshTarget":
        return cls("surface_axes_style", view_id)

    @classmethod
    def operator_overlay(cls, view_id: str) -> "RefreshTarget":
        return cls("operator_overlay", view_id)

    @classmethod
    def state_graph(cls, view_id: str) -> "RefreshTarget":
        return cls("state_graph", view_id)


RefreshTarget.CONTROLS = RefreshTarget.controls()


def _target_kind_counts(targets: set[RefreshTarget]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for target in targets:
        counts[target.kind] = counts.get(target.kind, 0) + 1
    return counts


class RefreshPlanner:
    SURFACE_VISUAL_PROPS = frozenset({"field_id", "geometry_id"})
    SURFACE_STYLE_PROPS = frozenset(
        {
            "color_map",
            "color_limits",
            "color_by",
            "surface_color",
            "surface_shading",
            "surface_alpha",
            "background_color",
        }
    )
    SURFACE_AXES_GEOMETRY_PROPS = frozenset(
        {
            "field_id",
            "geometry_id",
            "render_axes",
            "axes_in_middle",
            "tick_count",
            "tick_length_scale",
            "axis_labels",
        }
    )
    SURFACE_AXES_STYLE_PROPS = frozenset(
        {
            "tick_label_size",
            "axis_label_size",
            "axis_color",
            "text_color",
            "axis_alpha",
        }
    )
    GRID_SLICE_COMPUTE_PROPS = frozenset({"field_id", "geometry_id", "axis_state_key", "position_state_key"})
    GRID_SLICE_STYLE_PROPS = frozenset({"color", "alpha", "fill_alpha", "width"})
    LINE_PLOT_PROPS = frozenset(
        {
            "field_id",
            "operator_id",
            "x_dim",
            "series_dim",
            "selectors",
            "x_label",
            "y_label",
            "x_unit",
            "y_unit",
            "pen",
            "background_color",
            "title",
            "show_legend",
            "series_colors",
            "series_palette",
            "rolling_window",
            "trim_to_rolling_window",
            "max_refresh_hz",
            "y_min",
            "y_max",
            "x_major_tick_spacing",
            "x_minor_tick_spacing",
        }
    )

    def __init__(self, scene: Scene):
        self.scene = scene

    def _view_ids_in_3d_panels(self) -> tuple[str, ...]:
        return tuple(
            view_id
            for panel in self.scene.layout.panels_of_kind(PANEL_KIND_VIEW_3D)
            for view_id in panel.view_ids
        )

    def _view_3d(self, view_id: str):
        return self.scene.views.get(view_id)

    def morphology_views(self) -> dict[str, MorphologyViewSpec]:
        return {
            view_id: view
            for view_id in self._view_ids_in_3d_panels()
            if isinstance((view := self._view_3d(view_id)), MorphologyViewSpec)
        }

    def surface_views(self) -> dict[str, SurfaceViewSpec]:
        return {
            view_id: view
            for view_id in self._view_ids_in_3d_panels()
            if isinstance((view := self._view_3d(view_id)), SurfaceViewSpec)
        }

    def _panel_for_view(self, view_id: str, *, kind: str | None = None):
        return self.scene.layout.panel_for_view(view_id, kind=kind)

    def grid_slice_operators_for_view(self, view_id: str) -> dict[str, GridSliceOperatorSpec]:
        surface_view = self.surface_views().get(view_id)
        if surface_view is None:
            return {}
        panel = self._panel_for_view(view_id, kind=PANEL_KIND_VIEW_3D)
        if panel is None:
            return {}
        operators: dict[str, GridSliceOperatorSpec] = {}
        for operator_id in panel.operator_ids:
            operator = self.scene.operators.get(operator_id)
            if not isinstance(operator, GridSliceOperatorSpec):
                continue
            if operator.field_id != surface_view.field_id:
                continue
            if operator.geometry_id not in {None, surface_view.geometry_id}:
                continue
            operators[operator_id] = operator
        return operators

    def line_views(self) -> dict[str, LinePlotViewSpec]:
        return {
            view_id: view
            for panel in self.scene.layout.panels_of_kind(PANEL_KIND_LINE_PLOT)
            for view_id in panel.view_ids
            if isinstance((view := self.scene.views.get(view_id)), LinePlotViewSpec)
        }

    def state_graph_views(self) -> dict[str, StateGraphViewSpec]:
        return {
            view_id: view
            for panel in self.scene.layout.panels_of_kind(PANEL_KIND_STATE_GRAPH)
            for view_id in panel.view_ids
            if isinstance((view := self.scene.views.get(view_id)), StateGraphViewSpec)
        }

    def full_refresh_targets(self) -> set[RefreshTarget]:
        targets = {RefreshTarget.CONTROLS}
        for view_id in self.morphology_views():
            targets.add(RefreshTarget.morphology(view_id))
        for view_id in self.surface_views():
            targets.update(
                {
                    RefreshTarget.surface_visual(view_id),
                    RefreshTarget.surface_axes_geometry(view_id),
                    RefreshTarget.operator_overlay(view_id),
                }
            )
        for view_id in self.line_views():
            targets.add(RefreshTarget.line_plot(view_id))
        for view_id in self.state_graph_views():
            targets.add(RefreshTarget.state_graph(view_id))
        return targets

    def targets_for_state_change(self, state_key: str) -> set[RefreshTarget]:
        targets: set[RefreshTarget] = set()

        for view_id, morph_view in self.morphology_views().items():
            if binding_key(morph_view.background_color) == state_key or binding_key(morph_view.color_limits) == state_key:
                targets.add(RefreshTarget.morphology(view_id))

        for view_id, surface_view in self.surface_views().items():
            if any(binding_key(getattr(surface_view, prop)) == state_key for prop in self.SURFACE_VISUAL_PROPS if hasattr(surface_view, prop)):
                targets.add(RefreshTarget.surface_visual(view_id))
            if any(binding_key(getattr(surface_view, prop)) == state_key for prop in self.SURFACE_STYLE_PROPS if hasattr(surface_view, prop)):
                targets.add(RefreshTarget.surface_style(view_id))
            if any(
                binding_key(getattr(surface_view, prop)) == state_key
                for prop in self.SURFACE_AXES_GEOMETRY_PROPS
                if hasattr(surface_view, prop)
            ):
                targets.add(RefreshTarget.surface_axes_geometry(view_id))
            if any(
                binding_key(getattr(surface_view, prop)) == state_key
                for prop in self.SURFACE_AXES_STYLE_PROPS
                if hasattr(surface_view, prop)
            ):
                targets.add(RefreshTarget.surface_axes_style(view_id))
            operators = self.grid_slice_operators_for_view(view_id)
            if any(
                binding_key(getattr(operator, prop)) == state_key
                for operator in operators.values()
                for prop in self.GRID_SLICE_STYLE_PROPS
            ):
                targets.add(RefreshTarget.operator_overlay(view_id))
            if any(state_key in {operator.axis_state_key, operator.position_state_key} for operator in operators.values()):
                targets.add(RefreshTarget.operator_overlay(view_id))

        for view_id, line_view in self.line_views().items():
            if line_view.operator_id is not None:
                operator = self.scene.operators.get(line_view.operator_id)
                if isinstance(operator, GridSliceOperatorSpec) and state_key in {operator.axis_state_key, operator.position_state_key}:
                    targets.add(RefreshTarget.line_plot(view_id))
            if any(binding_key(value) == state_key for value in line_view.selectors.values()):
                targets.add(RefreshTarget.line_plot(view_id))
            if any(binding_key(getattr(line_view, prop)) == state_key for prop in ("pen", "background_color")):
                targets.add(RefreshTarget.line_plot(view_id))

        return targets

    def targets_for_field_replace(self, field_id: str, coords_changed: bool = True) -> set[RefreshTarget]:
        targets: set[RefreshTarget] = set()

        for view_id, morph_view in self.morphology_views().items():
            if morph_view.color_field_id == field_id:
                targets.add(RefreshTarget.morphology(view_id))

        for view_id, surface_view in self.surface_views().items():
            if surface_view.field_id != field_id:
                if any(operator.field_id == field_id for operator in self.grid_slice_operators_for_view(view_id).values()):
                    targets.add(RefreshTarget.operator_overlay(view_id))
                continue
            targets.add(RefreshTarget.surface_visual(view_id))
            if coords_changed or surface_view.color_limits is None:
                targets.add(RefreshTarget.surface_axes_geometry(view_id))
            if self.grid_slice_operators_for_view(view_id):
                targets.add(RefreshTarget.operator_overlay(view_id))

        for view_id, line_view in self.line_views().items():
            if line_view.operator_id is not None:
                operator = self.scene.operators.get(line_view.operator_id)
                if isinstance(operator, GridSliceOperatorSpec) and operator.field_id == field_id:
                    targets.add(RefreshTarget.line_plot(view_id))
            elif getattr(line_view, "field_id", None) == field_id:
                targets.add(RefreshTarget.line_plot(view_id))
        for view_id, state_graph_view in self.state_graph_views().items():
            if state_graph_view.node_field_id == field_id or state_graph_view.edge_field_id == field_id:
                targets.add(RefreshTarget.state_graph(view_id))
        return targets

    def targets_for_view_patch(self, view_id: str, changed_props: set[str]) -> set[RefreshTarget]:
        view = self.scene.views.get(view_id)
        if isinstance(view, MorphologyViewSpec):
            return {RefreshTarget.morphology(view_id)}
        if isinstance(view, SurfaceViewSpec):
            targets: set[RefreshTarget] = set()
            if changed_props & self.SURFACE_VISUAL_PROPS:
                targets.add(RefreshTarget.surface_visual(view_id))
            if changed_props & self.SURFACE_STYLE_PROPS:
                targets.add(RefreshTarget.surface_style(view_id))
            if changed_props & self.SURFACE_AXES_GEOMETRY_PROPS:
                targets.add(RefreshTarget.surface_axes_geometry(view_id))
            if changed_props & self.SURFACE_AXES_STYLE_PROPS:
                targets.add(RefreshTarget.surface_axes_style(view_id))
            if changed_props & {"field_id", "geometry_id"}:
                targets.add(RefreshTarget.operator_overlay(view_id))
            if "max_refresh_hz" in changed_props:
                targets.add(RefreshTarget.surface_visual(view_id))
            return targets
        if isinstance(view, LinePlotViewSpec) and changed_props & self.LINE_PLOT_PROPS:
            return {RefreshTarget.line_plot(view_id)}
        if isinstance(view, StateGraphViewSpec):
            return {RefreshTarget.state_graph(view_id)}
        return set()

    def targets_for_operator_patch(self, operator_id: str, changed_props: set[str]) -> set[RefreshTarget]:
        targets: set[RefreshTarget] = set()
        for view_id in self.surface_views():
            if operator_id in self.grid_slice_operators_for_view(view_id):
                targets.add(RefreshTarget.operator_overlay(view_id))
        for view_id, line_view in self.line_views().items():
            if line_view.operator_id == operator_id and changed_props & self.GRID_SLICE_COMPUTE_PROPS:
                targets.add(RefreshTarget.line_plot(view_id))
        return targets


def resolve_value(value, state: dict[str, Any]):
    if isinstance(value, StateBinding):
        return state.get(value.key)
    return value


def binding_key(value) -> str | None:
    if isinstance(value, StateBinding):
        return value.key
    return None
