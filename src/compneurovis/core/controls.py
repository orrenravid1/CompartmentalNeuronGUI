from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compneurovis.core.bindings import AttributeRef


@dataclass(frozen=True, slots=True)
class ControlSpec:
    id: str
    kind: str
    label: str
    default: Any
    min: float | int | None = None
    max: float | int | None = None
    steps: int | None = None
    options: tuple[str, ...] = ()
    state_key: str | None = None
    send_to_session: bool = False
    scale: str = "linear"
    target: AttributeRef | None = None

    def resolved_state_key(self) -> str:
        return self.state_key or self.id


@dataclass(frozen=True, slots=True)
class XYControlSpec:
    x_id: str
    y_id: str
    label: str = ""
    x_label: str = "X"
    y_label: str = "Y"
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    x_default: float = 0.5
    y_default: float = 0.5
    x_state_key: str | None = None
    y_state_key: str | None = None
    shape: str = "square"
    send_to_session: bool = False

    def resolved_x_state_key(self) -> str:
        return self.x_state_key or self.x_id

    def resolved_y_state_key(self) -> str:
        return self.y_state_key or self.y_id


@dataclass(frozen=True, slots=True)
class ActionSpec:
    id: str
    label: str
    payload: dict[str, Any] = field(default_factory=dict)
    shortcuts: tuple[str, ...] = ()
    selection_mode: bool = False
    selection_payload_key: str = "entity_id"
