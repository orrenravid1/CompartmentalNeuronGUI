from __future__ import annotations

import numpy as np

from compneurovis.core import AppSpec, ControlSpec, Document, Field, GridGeometry, LayoutSpec, LinePlotViewSpec, SurfaceViewSpec


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
    controls: dict[str, ControlSpec] | None = None,
) -> AppSpec:
    if surface_view is None:
        surface_view = SurfaceViewSpec(
            id="surface",
            title=title,
            field_id=field.id,
            geometry_id=geometry.id if geometry is not None else None,
        )
    views = {surface_view.id: surface_view}
    layout = LayoutSpec(title=title, view_3d_ids=(surface_view.id,))
    if line_view is not None:
        views[line_view.id] = line_view
        layout.line_plot_view_id = line_view.id
    controls = {} if controls is None else dict(controls)
    layout.control_ids = tuple(controls.keys())
    document = Document(
        fields={field.id: field},
        geometries={} if geometry is None else {geometry.id: geometry},
        views=views,
        controls=controls,
        layout=layout,
    )
    return AppSpec(document=document, title=title)
