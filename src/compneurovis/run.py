from __future__ import annotations

import multiprocessing as mp

from compneurovis.core.app import RunSpec
from compneurovis.hosts import configure_diagnostics, configure_multiprocessing


def run_app(run_spec: RunSpec) -> None:
    # Guard against re-entry in spawned worker processes (Windows spawn model).
    if mp.current_process().name != "MainProcess":
        return
    configure_multiprocessing()
    configure_diagnostics(run_spec.diagnostics)

    app_spec = run_spec.app_spec
    if app_spec is None:
        raise ValueError("RunSpec.app_spec is required.")

    endpoints = run_spec.transport(run_spec.actors) if run_spec.transport is not None else {}

    startables = []
    for spec in run_spec.actors:
        endpoint = endpoints.get(spec.id)
        startables.append(spec.host_source(app_spec, endpoint))

    for s in startables:
        s.start()
    try:
        for s in startables:
            s.run()
    finally:
        for s in reversed(startables):
            s.stop()


__all__ = ["run_app"]
