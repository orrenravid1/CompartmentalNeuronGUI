from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from compneurovis.core.scene import PanelSpec, Scene


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
class KeyPressed(SessionCommand):
    key: str


@dataclass(frozen=True, slots=True)
class EntityClicked(SessionCommand):
    entity_id: str


@dataclass(frozen=True, slots=True)
class StopSession(SessionCommand):
    pass


@dataclass(frozen=True, slots=True)
class SessionUpdate:
    pass


@dataclass(frozen=True, slots=True)
class SceneReady(SessionUpdate):
    scene: Scene


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
class ScenePatch(SessionUpdate):
    view_updates: dict[str, dict[str, Any]] = field(default_factory=dict)
    operator_updates: dict[str, dict[str, Any]] = field(default_factory=dict)
    control_updates: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata_updates: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StatePatch(SessionUpdate):
    updates: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PanelPatch(SessionUpdate):
    """Surgical update to one panel's contents. Does not affect other panels or scene data.

    Fields set to ``None`` are left unchanged. Use an empty tuple to explicitly clear a list.
    Only ``kind="controls"`` panels support ``control_ids`` / ``action_ids`` updates.
    For structural panel changes (kind, camera settings, add/remove panels) use ``LayoutReplace``.
    """

    panel_id: str
    control_ids: tuple[str, ...] | None = None
    action_ids: tuple[str, ...] | None = None
    view_ids: tuple[str, ...] | None = None
    title: str | None = None


@dataclass(frozen=True, slots=True)
class LayoutReplace(SessionUpdate):
    """Replace the full panel arrangement without rebuilding scene data.

    Replaces ``LayoutSpec.panels`` and ``panel_grid`` on the frontend scene. Fields,
    geometries, views, operators, controls, and actions are untouched. The frontend
    rebuilds the widget tree and triggers a full content refresh for the new panels.

    Pass ``panel_grid=()`` to use auto-layout.
    """

    panels: tuple[PanelSpec, ...]
    panel_grid: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class Status(SessionUpdate):
    message: str
    timeout_ms: int | None = None


@dataclass(frozen=True, slots=True)
class Error(SessionUpdate):
    message: str
