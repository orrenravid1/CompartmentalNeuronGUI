from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def _coerce_coord(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError("Field coordinates must be one-dimensional")
    return arr


@dataclass(frozen=True, slots=True)
class Field:
    """Dense labeled array with named axes and coordinate metadata."""

    id: str
    values: np.ndarray
    dims: tuple[str, ...]
    coords: dict[str, np.ndarray]
    unit: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        dims = tuple(self.dims)
        coords = {str(name): _coerce_coord(coord) for name, coord in self.coords.items()}

        if values.ndim != len(dims):
            raise ValueError(
                f"Field '{self.id}' has {values.ndim} dimensions but dims={dims}"
            )
        if set(coords.keys()) != set(dims):
            raise ValueError(
                f"Field '{self.id}' coords keys must exactly match dims {dims}"
            )
        for axis, dim in enumerate(dims):
            if len(coords[dim]) != values.shape[axis]:
                raise ValueError(
                    f"Field '{self.id}' coord '{dim}' has length {len(coords[dim])}, "
                    f"expected {values.shape[axis]}"
                )

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "attrs", dict(self.attrs))

    def axis_index(self, dim: str) -> int:
        try:
            return self.dims.index(dim)
        except ValueError as exc:
            raise KeyError(f"Unknown dimension '{dim}' for field '{self.id}'") from exc

    def coord(self, dim: str) -> np.ndarray:
        return self.coords[dim]

    def with_values(
        self,
        values: np.ndarray,
        coords: Mapping[str, np.ndarray] | None = None,
        attrs_update: Mapping[str, Any] | None = None,
    ) -> Field:
        merged_attrs = dict(self.attrs)
        if attrs_update:
            merged_attrs.update(attrs_update)
        return Field(
            id=self.id,
            values=np.asarray(values),
            dims=self.dims,
            coords=dict(self.coords if coords is None else coords),
            unit=self.unit,
            attrs=merged_attrs,
        )

    def resolve_indexer(self, dim: str, selector: Any) -> int | slice:
        coord = self.coord(dim)
        if isinstance(selector, slice):
            return selector
        if isinstance(selector, (int, np.integer)):
            return int(selector)
        if isinstance(selector, str):
            matches = np.where(coord.astype(str) == selector)[0]
            if not len(matches):
                raise KeyError(
                    f"Field '{self.id}' coord '{dim}' does not contain label '{selector}'"
                )
            return int(matches[0])
        if isinstance(selector, (float, np.floating)):
            if not np.issubdtype(coord.dtype, np.number):
                raise TypeError(
                    f"Field '{self.id}' coord '{dim}' is not numeric, cannot resolve float selector"
                )
            return int(np.argmin(np.abs(coord.astype(float) - float(selector))))
        raise TypeError(
            f"Unsupported selector type {type(selector)!r} for field '{self.id}' dim '{dim}'"
        )

    def select(self, selectors: Mapping[str, Any]) -> Field:
        indexers: list[Any] = [slice(None)] * len(self.dims)
        remaining_dims: list[str] = list(self.dims)
        remaining_coords = dict(self.coords)

        for dim, selector in selectors.items():
            axis = self.axis_index(dim)
            indexer = self.resolve_indexer(dim, selector)
            indexers[axis] = indexer
            if not isinstance(indexer, slice):
                remaining_dims.remove(dim)
                remaining_coords.pop(dim, None)

        values = self.values[tuple(indexers)]

        for dim, indexer in selectors.items():
            axis = self.axis_index(dim)
            if isinstance(indexer, slice):
                remaining_coords[dim] = remaining_coords[dim][indexer]

        ordered_coords = {dim: remaining_coords[dim] for dim in remaining_dims}
        return Field(
            id=self.id,
            values=np.asarray(values),
            dims=tuple(remaining_dims),
            coords=ordered_coords,
            unit=self.unit,
            attrs=dict(self.attrs),
        )

