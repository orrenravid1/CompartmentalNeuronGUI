from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ValueOrBinding = Any


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    id: str


@dataclass(frozen=True, slots=True)
class GridSliceOperatorSpec(OperatorSpec):
    field_id: str = ""
    geometry_id: str | None = None
    axis_state_key: str | None = None
    position_state_key: str | None = None
    color: ValueOrBinding = "#111111"
    alpha: ValueOrBinding = 0.95
    fill_alpha: ValueOrBinding = 0.0
    width: ValueOrBinding = 3.0
