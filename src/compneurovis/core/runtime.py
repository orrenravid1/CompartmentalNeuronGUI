from __future__ import annotations

import threading
import time
from typing import Any

from compneurovis.core.app import AppSpec, DiagnosticsSpec


class AppRuntime:
    """Authoritative coordinator for a single app run.

    Owns the startup contract (AppSpec), the stop signal, and the
    foreground/background threading policy. Does not own simulation state,
    renderer state, transport endpoints, or actor construction logic.
    """

    def __init__(
        self,
        *,
        app_spec: AppSpec,
        diagnostics: DiagnosticsSpec | None = None,
    ) -> None:
        self._app_spec = app_spec
        self._diagnostics = diagnostics
        self._stop_event = threading.Event()

    @property
    def app_spec(self) -> AppSpec:
        return self._app_spec

    @property
    def diagnostics(self) -> DiagnosticsSpec | None:
        return self._diagnostics

    def stop(self) -> None:
        self._stop_event.set()

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def wait(self, items: list[tuple[Any, Any]]) -> None:
        """Run all startables, blocking until they all finish.

        Startables with spec.runs_in_foreground=True run in the calling (main)
        thread. All others run as daemon threads. At most one foreground
        startable is expected per run.
        """
        foreground = [(spec, s) for spec, s in items if spec.runs_in_foreground]
        background = [(spec, s) for spec, s in items if not spec.runs_in_foreground]

        threads = [threading.Thread(target=s.run, daemon=True) for _, s in background]
        for t in threads:
            t.start()

        if foreground:
            _, fg = foreground[0]
            fg.run()  # blocks until event loop exits
            self.stop()
        else:
            # Headless or all-remote: poll until stop() is signalled or all threads finish.
            while not self.is_stopped() and any(t.is_alive() for t in threads):
                time.sleep(0.05)
            self.stop()

        for t in threads:
            t.join(timeout=5.0)


__all__ = ["AppRuntime"]
