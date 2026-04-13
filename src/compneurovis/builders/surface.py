from __future__ import annotations

import numpy as np

from compneurovis.core import (
    AppSpec,
    ControlSpec,
    Field,
    GridSliceOperatorSpec,
    GridGeometry,
    LayoutSpec,
    LinePlotViewSpec,
    OperatorSpec,
    Scene,
    SurfaceViewSpec,
    View3DHostSpec,
)


def grid_field(
    *,
    field_id: str,
    values: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    x_dim: str = "x",
    y_dim: str = "y",
    unit: str | None = None,
) -> tuple[Field, GridGeometry]:
    """Create a Field/GridGeometry pair from a 2-D array and coordinate vectors."""

    field = Field(
        id=field_id,
        values=np.asarray(values, dtype=np.float32),
        dims=(y_dim, x_dim),
        coords={
            y_dim: np.asarray(y_coords, dtype=np.float32),
            x_dim: np.asarray(x_coords, dtype=np.float32),
        },
        unit=unit,
    )
    geometry = GridGeometry(
        id=f"{field_id}-grid",
        dims=(y_dim, x_dim),
        coords={
            y_dim: np.asarray(y_coords, dtype=np.float32),
            x_dim: np.asarray(x_coords, dtype=np.float32),
        },
    )
    return field, geometry


def build_surface_app(
    *,
    field: Field,
    geometry: GridGeometry | None = None,
    title: str = "Surface",
    surface_view: SurfaceViewSpec | None = None,
    line_view: LinePlotViewSpec | None = None,
    operators: dict[str, OperatorSpec] | None = None,
    controls: dict[str, ControlSpec] | None = None,
    view_3d_host: View3DHostSpec | None = None,
) -> AppSpec:
    """Build a static surface app from field data, optional geometry, and views.

    Pass ``view_3d_host`` to override host-level camera settings such as the
    initial turntable distance for the surface viewport.
    """

    if surface_view is None:
        surface_view = SurfaceViewSpec(
            id="surface",
            title=title,
            field_id=field.id,
            geometry_id=geometry.id if geometry is not None else None,
        )
    views = {surface_view.id: surface_view}
    layout = LayoutSpec(title=title, view_3d_ids=(surface_view.id,))
    default_operator_ids = tuple(
        operator.id
        for operator in (operators or {}).values()
        if isinstance(operator, GridSliceOperatorSpec)
        and operator.field_id == field.id
        and (geometry is None or operator.geometry_id in {None, geometry.id})
    )
    if view_3d_host is not None:
        if not view_3d_host.operator_ids and default_operator_ids:
            view_3d_host = View3DHostSpec(
                id=view_3d_host.id,
                view_ids=view_3d_host.view_ids,
                operator_ids=default_operator_ids,
                kind=view_3d_host.kind,
                title=view_3d_host.title,
                camera_distance=view_3d_host.camera_distance,
                camera_elevation=view_3d_host.camera_elevation,
                camera_azimuth=view_3d_host.camera_azimuth,
            )
        layout.view_3d_hosts = (view_3d_host,)
    else:
        if default_operator_ids:
            layout.view_3d_hosts = (View3DHostSpec(id=surface_view.id, view_ids=(surface_view.id,), operator_ids=default_operator_ids),)
    if line_view is not None:
        views[line_view.id] = line_view
        layout.line_plot_view_id = line_view.id
    operators = {} if operators is None else dict(operators)
    controls = {} if controls is None else dict(controls)
    layout.control_ids = tuple(controls.keys())
    scene = Scene(
        fields={field.id: field},
        geometries={} if geometry is None else {geometry.id: geometry},
        views=views,
        operators=operators,
        controls=controls,
        layout=layout,
    )
    return AppSpec(scene=scene, title=title)
