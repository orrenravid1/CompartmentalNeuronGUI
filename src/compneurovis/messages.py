from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from compneurovis.core.app import AppSpec, PanelSpec


@dataclass(frozen=True, slots=True)
class CommandPayload:
    pass


@dataclass(frozen=True, slots=True)
class Reset(CommandPayload):
    pass


@dataclass(frozen=True, slots=True)
class SetControl(CommandPayload):
    control_id: str
    value: Any


@dataclass(frozen=True, slots=True)
class InvokeAction(CommandPayload):
    action_id: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class KeyPressed(CommandPayload):
    key: str


@dataclass(frozen=True, slots=True)
class EntityClicked(CommandPayload):
    entity_id: str


@dataclass(frozen=True, slots=True)
class StopBackend(CommandPayload):
    pass


@dataclass(frozen=True, slots=True)
class UpdatePayload:
    pass


@dataclass(frozen=True, slots=True)
class AppSpecReady(UpdatePayload):
    app_spec: AppSpec


@dataclass(frozen=True, slots=True)
class FieldReplace(UpdatePayload):
    field_id: str
    values: np.ndarray
    coords: dict[str, np.ndarray] | None = None
    attrs_update: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FieldAppend(UpdatePayload):
    field_id: str
    append_dim: str
    values: np.ndarray
    coord_values: np.ndarray
    max_length: int | None = None
    attrs_update: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AppSpecPatch(UpdatePayload):
    view_updates: dict[str, dict[str, Any]] = field(default_factory=dict)
    operator_updates: dict[str, dict[str, Any]] = field(default_factory=dict)
    control_updates: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata_updates: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StatePatch(UpdatePayload):
    updates: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PanelPatch(UpdatePayload):
    """Surgical update to one panel's contents. Does not affect other panels or AppSpec data.

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
class LayoutReplace(UpdatePayload):
    """Replace the full panel arrangement without rebuilding AppSpec data.

    Replaces ``LayoutSpec.panels`` and ``panel_grid`` on the frontend AppSpec. Fields,
    geometries, views, operators, controls, and actions are untouched. The frontend
    rebuilds the widget tree and triggers a full content refresh for the new panels.

    Pass ``panel_grid=()`` to use auto-layout.
    """

    panels: tuple[PanelSpec, ...]
    panel_grid: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class Status(UpdatePayload):
    message: str
    timeout_ms: int | None = None


@dataclass(frozen=True, slots=True)
class Error(UpdatePayload):
    message: str
