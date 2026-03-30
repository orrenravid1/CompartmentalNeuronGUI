from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AttributeRef:
    owner: str
    attribute: str

    def read(self, root: Any) -> Any:
        return getattr(getattr(root, self.owner), self.attribute)

    def write(self, root: Any, value: Any) -> None:
        setattr(getattr(root, self.owner), self.attribute, value)


@dataclass(frozen=True, slots=True)
class SeriesSpec:
    key: str
    label: str
    source: AttributeRef
    color: Any = "k"
