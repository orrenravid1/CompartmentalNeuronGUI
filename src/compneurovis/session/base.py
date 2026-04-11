from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Callable, Deque, TypeAlias

from compneurovis.core.scene import Scene
from compneurovis.session.protocol import SessionCommand, SessionUpdate


class Session(ABC):
    @classmethod
    def startup_scene(cls) -> Scene | None:
        return None

    @abstractmethod
    def initialize(self) -> Scene | None:
        pass

    @abstractmethod
    def advance(self) -> None:
        pass

    @abstractmethod
    def handle(self, command: SessionCommand) -> None:
        pass

    @abstractmethod
    def read_updates(self) -> list[SessionUpdate]:
        pass

    def is_live(self) -> bool:
        return True

    def idle_sleep(self) -> float:
        return 0.05

    def shutdown(self) -> None:
        pass


class BufferedSession(Session):
    def __init__(self) -> None:
        self._updates: Deque[SessionUpdate] = deque()

    def emit(self, update: SessionUpdate) -> None:
        self._updates.append(update)

    def read_updates(self) -> list[SessionUpdate]:
        updates = list(self._updates)
        self._updates.clear()
        return updates


SessionFactory: TypeAlias = Callable[[], Session]
SessionSource: TypeAlias = type[Session] | SessionFactory


def resolve_startup_scene_source(source: SessionSource | None) -> Scene | None:
    if source is None:
        return None
    if isinstance(source, type):
        if not issubclass(source, Session):
            raise TypeError(f"Expected Session subclass, got {source!r}")
        scene = source.startup_scene()
    else:
        if isinstance(source, Session):
            raise TypeError(
                "Eager session instances are not supported for worker-backed apps. "
                "Pass a Session subclass or a top-level zero-argument factory instead."
            )
        bootstrap = getattr(source, "startup_scene", None)
        if not callable(bootstrap):
            return None
        scene = bootstrap()
    if scene is not None and not isinstance(scene, Scene):
        raise TypeError(f"Startup scene source returned {type(scene)!r}, expected Scene | None")
    return scene


def resolve_session_source(source: SessionSource) -> Session:
    if isinstance(source, type):
        if not issubclass(source, Session):
            raise TypeError(f"Expected Session subclass, got {source!r}")
        return source()
    if isinstance(source, Session):
        raise TypeError(
            "Eager session instances are not supported for worker-backed apps. "
            "Pass a Session subclass or a top-level zero-argument factory instead."
        )
    if callable(source):
        session = source()
        if not isinstance(session, Session):
            raise TypeError(f"Session factory returned {type(session)!r}, expected Session")
        return session
    raise TypeError(f"Unsupported session source: {source!r}")


def resolve_interaction_target_source(source: Any | None) -> Any | None:
    if source is None:
        return None
    if isinstance(source, type):
        return source()
    if callable(source) and not any(hasattr(source, attr) for attr in ("on_action", "on_key_press", "on_entity_clicked")):
        return source()
    return source
