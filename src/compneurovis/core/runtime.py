from __future__ import annotations

import threading

from compneurovis.core.app import AppSpec, DiagnosticsSpec


class AppRuntime:
    """Authoritative coordinator for a single app run.

    Owns the startup contract (AppSpec) and the stop signal. The
    foreground/background launch lifecycle is the single implementation in
    AppHandle.wait(). Does not own simulation state, renderer state,
    transport endpoints, or actor construction logic.
    """

    def __init__(
        self,
        *,
        app_spec: AppSpec | None,
        diagnostics: DiagnosticsSpec | None = None,
    ) -> None:
        self._app_spec = app_spec
        self._diagnostics = diagnostics
        self._stop_event = threading.Event()

    @property
    def app_spec(self) -> AppSpec | None:
        """The authoritative startup blueprint — read-only after construction.

        May be None: in the multiprocess desktop path the backend is
        authoritative and announces the AppSpec via AppSpecSnapshot, so the
        main process no longer builds it. Actors must not mutate this object.
        Each actor derives its own mutable working state (frontend: AppState)
        by deep-copying this seed and folding the routed patch stream into the
        copy. The copy boundary is what keeps this object authoritative; no
        in-process mutator path to it remains.
        """
        return self._app_spec

    @property
    def diagnostics(self) -> DiagnosticsSpec | None:
        return self._diagnostics

    def stop(self) -> None:
        self._stop_event.set()

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()


__all__ = ["AppRuntime"]
