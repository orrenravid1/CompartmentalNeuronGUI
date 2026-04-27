from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from compneurovis.core.field import Field
from compneurovis.core.geometry import GridGeometry
from compneurovis.core.operators import GridSliceOperatorSpec
from compneurovis.core.state import StateBinding


@dataclass(slots=True)
class SurfaceSceneData:
    field_id: str
    x_dim: str
    y_dim: str
    x_grid: np.ndarray
    y_grid: np.ndarray
    z: np.ndarray
    coords: dict[str, np.ndarray]


def resolve_binding(value, state: dict[str, Any]):
    if isinstance(value, StateBinding):
        return state.get(value.key)
    return value


def surface_scene_from_field(field: Field, geometry: GridGeometry | None) -> SurfaceSceneData:
    if geometry is None:
        y_dim, x_dim = field.dims[0], field.dims[1]
        y_coords = field.coord(y_dim)
        x_coords = field.coord(x_dim)
    else:
        y_dim, x_dim = geometry.dims[0], geometry.dims[1]
        y_coords = geometry.coords[y_dim]
        x_coords = geometry.coords[x_dim]

    if field.dims != (y_dim, x_dim):
        axis_map = {dim: idx for idx, dim in enumerate(field.dims)}
        z = np.transpose(field.values, axes=(axis_map[y_dim], axis_map[x_dim]))
    else:
        z = field.values

    x_grid, y_grid = np.meshgrid(np.asarray(x_coords, dtype=np.float32), np.asarray(y_coords, dtype=np.float32))
    return SurfaceSceneData(
        field_id=field.id,
        x_dim=x_dim,
        y_dim=y_dim,
        x_grid=x_grid,
        y_grid=y_grid,
        z=np.asarray(z, dtype=np.float32),
        coords={
            x_dim: np.asarray(x_coords, dtype=np.float32),
            y_dim: np.asarray(y_coords, dtype=np.float32),
        },
    )


def resolve_grid_slice_position(
    coords: dict[str, np.ndarray],
    *,
    axis_state_key: str | None,
    position_state_key: str | None,
    state: dict[str, Any],
    default_axis: str,
):
    if not axis_state_key or not position_state_key:
        return None
    axis = state.get(axis_state_key, default_axis)
    if axis not in coords:
        axis = default_axis if default_axis in coords else next(iter(coords))
    normalized = min(1.0, max(0.0, float(state.get(position_state_key, 0.0))))
    axis_coords = np.asarray(coords[axis], dtype=np.float32)
    idx = max(0, min(len(axis_coords) - 1, int(round(normalized * (len(axis_coords) - 1)))))
    return axis, idx, float(axis_coords[idx])


def overlay_from_grid_slice_operator(
    surface_scene: SurfaceSceneData,
    operator: GridSliceOperatorSpec,
    resolved_state: dict[str, Any],
):
    resolved = resolve_grid_slice_position(
        surface_scene.coords,
        axis_state_key=operator.axis_state_key,
        position_state_key=operator.position_state_key,
        state=resolved_state,
        default_axis=surface_scene.x_dim,
    )
    if resolved is None:
        return None
    axis, _idx, value = resolved
    return {
        "operator_id": operator.id,
        "axis": "x" if axis == surface_scene.x_dim else "y",
        "value": value,
        "color": resolved_state[f"{operator.id}:color"],
        "alpha": resolved_state[f"{operator.id}:alpha"],
        "fill_alpha": resolved_state[f"{operator.id}:fill_alpha"],
        "width": resolved_state[f"{operator.id}:width"],
    }


def line_from_grid_slice_operator(field: Field, operator: GridSliceOperatorSpec, state: dict[str, Any]):
    if field.values.ndim != 2:
        raise ValueError("grid slice operators require a 2D field")
    resolved = resolve_grid_slice_position(
        {dim: field.coord(dim) for dim in field.dims},
        axis_state_key=operator.axis_state_key,
        position_state_key=operator.position_state_key,
        state=state,
        default_axis=field.dims[-1],
    )
    if resolved is None:
        return None
    slice_dim, idx, slice_value = resolved
    other_dims = [dim for dim in field.dims if dim != slice_dim]
    if len(other_dims) != 1:
        raise ValueError("grid slice operators require exactly one non-sliced dimension")
    x_dim = other_dims[0]
    sliced = field.select({slice_dim: idx})
    return (
        np.asarray(sliced.coord(x_dim), dtype=np.float32),
        np.asarray(sliced.values, dtype=np.float32),
        x_dim,
        slice_dim,
        slice_value,
    )


def field_from_grid_slice_operator(
    field: Field, operator: GridSliceOperatorSpec, state: dict[str, Any]
) -> Field | None:
    result = line_from_grid_slice_operator(field, operator, state)
    if result is None:
        return None
    x, y, x_dim, slice_dim, slice_value = result
    return Field(
        id=f"{field.id} at {slice_dim}={slice_value:.3f}",
        values=y,
        dims=(x_dim,),
        coords={x_dim: x},
    )
