from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from compneurovis.core.document import Document


@dataclass(frozen=True, slots=True)
class SessionCommand:
    pass


@dataclass(frozen=True, slots=True)
class Reset(SessionCommand):
    pass


@dataclass(frozen=True, slots=True)
class SetControl(SessionCommand):
    control_id: str
    value: Any


@dataclass(frozen=True, slots=True)
class InvokeAction(SessionCommand):
    action_id: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StopSession(SessionCommand):
    pass


@dataclass(frozen=True, slots=True)
class SessionUpdate:
    pass


@dataclass(frozen=True, slots=True)
class DocumentReady(SessionUpdate):
    document: Document


@dataclass(frozen=True, slots=True)
class FieldReplace(SessionUpdate):
    field_id: str
    values: np.ndarray
    coords: dict[str, np.ndarray] | None = None
    attrs_update: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FieldAppend(SessionUpdate):
    field_id: str
    append_dim: str
    values: np.ndarray
    coord_values: np.ndarray
    max_length: int | None = None
    attrs_update: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DocumentPatch(SessionUpdate):
    view_updates: dict[str, dict[str, Any]] = field(default_factory=dict)
    control_updates: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata_updates: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Status(SessionUpdate):
    message: str


@dataclass(frozen=True, slots=True)
class Error(SessionUpdate):
    message: str
