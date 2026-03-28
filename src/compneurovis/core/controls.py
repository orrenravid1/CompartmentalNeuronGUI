from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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

    def resolved_state_key(self) -> str:
        return self.state_key or self.id


@dataclass(frozen=True, slots=True)
class ActionSpec:
    id: str
    label: str
    payload: dict[str, Any] = field(default_factory=dict)

