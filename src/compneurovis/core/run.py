from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING, Any

from compneurovis.core.app import ActorSpec, RunSpec
from compneurovis.core.hosts import ConnectionSlotHost, configure_diagnostics, configure_multiprocessing
from compneurovis.core.runtime import AppRuntime

if TYPE_CHECKING:
    from compneurovis.core.actor import ActorSource


def run_app(run_spec: RunSpec) -> None:
    """Bundled launch: orchestrator + all actors started from this process.

    This is the convenient single-script entry point. Internally equivalent to
    run_orchestrator + run_as_backend/run_as_frontend in separate processes.
    """
    if mp.current_process().name != "MainProcess":
        return
    configure_multiprocessing()

    if run_spec.app_spec is None:
        raise ValueError("RunSpec.app_spec is required.")

    runtime = AppRuntime(app_spec=run_spec.app_spec, diagnostics=run_spec.diagnostics)
    configure_diagnostics(runtime.diagnostics)

    fg_actors = [s for s in run_spec.actors if s.runs_in_foreground]
    if len(fg_actors) > 1:
        raise ValueError(
            f"At most one foreground actor allowed; got {[s.id for s in fg_actors]}."
        )

    endpoints = run_spec.transport(run_spec.actors) if run_spec.transport is not None else {}
    items: list[tuple[ActorSpec, Any]] = []
    for spec in run_spec.actors:
        endpoint = endpoints.get(spec.id)
        if spec.host_source is None:
            items.append((spec, ConnectionSlotHost(endpoint)))
        else:
            items.append((spec, spec.host_source(runtime, endpoint)))

    for _, s in items:
        s.start()
    try:
        runtime.wait(items)
    finally:
        for _, s in reversed(items):
            s.stop()


def run_orchestrator(run_spec: RunSpec) -> None:
    """Pure orchestrator: transport fabric + AppSpec authority + lifecycle.

    All ActorSpec.host_source must be None — actors connect independently via
    their own run_as_backend / run_as_frontend calls. Use run_app when all
    actors are launched from the same process.
    """
    if mp.current_process().name != "MainProcess":
        return

    hosted = [s for s in run_spec.actors if s.host_source is not None]
    if hosted:
        raise ValueError(
            f"run_orchestrator expects all host_source=None; use run_app for bundled launch. "
            f"Hosted actors: {[s.id for s in hosted]}"
        )

    configure_multiprocessing()

    if run_spec.app_spec is None:
        raise ValueError("RunSpec.app_spec is required.")

    runtime = AppRuntime(app_spec=run_spec.app_spec, diagnostics=run_spec.diagnostics)
    configure_diagnostics(runtime.diagnostics)

    endpoints = run_spec.transport(run_spec.actors) if run_spec.transport is not None else {}
    items: list[tuple[ActorSpec, Any]] = [
        (spec, ConnectionSlotHost(endpoints.get(spec.id)))
        for spec in run_spec.actors
    ]

    for _, s in items:
        s.start()
    try:
        runtime.wait(items)
    finally:
        for _, s in reversed(items):
            s.stop()


def run_as_backend(_backend_source: ActorSource, _url: str) -> None:
    """Connect to an existing orchestrator and run as the backend actor.

    Receives AppSpec over the startup channel. Does not construct AppRuntime
    or RunSpec — this process is a client, not a coordinator.

    Requires a WebSocket-capable transport; not yet implemented.
    """
    raise NotImplementedError(
        "run_as_backend requires a WebSocket transport which is not yet implemented."
    )


def run_as_frontend(_frontend_source: ActorSource, _url: str) -> None:
    """Connect to an existing orchestrator and run as the frontend actor.

    Receives AppSpec over the startup channel. Does not construct AppRuntime
    or RunSpec — this process is a client, not a coordinator.

    Requires a WebSocket-capable transport; not yet implemented.
    """
    raise NotImplementedError(
        "run_as_frontend requires a WebSocket transport which is not yet implemented."
    )


__all__ = ["run_app", "run_orchestrator", "run_as_backend", "run_as_frontend"]
