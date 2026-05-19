from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING, Any

from compneurovis.core.app import ActorSpec, RunSpec
from compneurovis.core.hosts import AppHandle, ConnectionSlotHost, configure_diagnostics, configure_multiprocessing
from compneurovis.core.runtime import AppRuntime

if TYPE_CHECKING:
    from compneurovis.core.actor import ActorSource


def run_app(run_spec: RunSpec) -> None:
    """Bundled launch: orchestrator + all actors started from this process.

    Blocks until the foreground actor (e.g. Qt event loop) exits.
    Guard against subprocess re-imports — safe to call unconditionally in scripts.
    """
    if mp.current_process().name != "MainProcess":
        return
    start_app(run_spec).wait()


def run_orchestrator(run_spec: RunSpec) -> None:
    """Pure orchestrator: transport fabric + AppSpec authority + lifecycle.

    All ActorSpec.host_source must be None — actors connect independently via
    their own run_as_backend / run_as_frontend calls. Sugar over
    start_app().wait(); use run_app when all actors launch from this process.
    """
    if mp.current_process().name != "MainProcess":
        return

    hosted = [s for s in run_spec.actors if s.host_source is not None]
    if hosted:
        raise ValueError(
            f"run_orchestrator expects all host_source=None; use run_app for bundled launch. "
            f"Hosted actors: {[s.id for s in hosted]}"
        )

    start_app(run_spec).wait()


def start_app(run_spec: RunSpec) -> AppHandle:
    """Universal launch — starts all actors and returns an AppHandle.

    Non-foreground actors have run() called immediately (subprocess no-op,
    thread start, asyncio task). Foreground actors (e.g. Qt) are deferred to
    AppHandle.wait(), which must be called from the main thread.

    For notebooks: no foreground actor — widget is in AppHandle.results.
    For desktop: call AppHandle.wait() (or use run_app which does this).
    """
    # app_spec may be None: the backend can be authoritative and announce it
    # at startup via AppSpecSnapshot (multiprocess desktop path). Frontends
    # start in a loading state until the snapshot arrives.
    configure_multiprocessing()

    fg_actors = [s for s in run_spec.actors if s.runs_in_foreground]
    if len(fg_actors) > 1:
        raise ValueError(
            f"At most one foreground actor allowed; got {[s.id for s in fg_actors]}."
        )

    runtime = AppRuntime(app_spec=run_spec.app_spec, diagnostics=run_spec.diagnostics)
    configure_diagnostics(runtime.diagnostics)

    endpoints = run_spec.transport(run_spec.actors) if run_spec.transport is not None else {}
    items: list[tuple[ActorSpec, Any]] = []
    for spec in run_spec.actors:
        endpoint = endpoints.get(spec.id)
        if spec.host_source is None:
            items.append((spec, ConnectionSlotHost(endpoint)))
        else:
            items.append((spec, spec.host_source(runtime, endpoint)))

    for _, host in items:
        host.start()

    results: dict[str, Any] = {}
    for spec, host in items:
        if not spec.runs_in_foreground:
            result = host.run()
            if result is not None:
                results[spec.id] = result

    return AppHandle(runtime=runtime, items=items, results=results)


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


__all__ = ["run_app", "run_orchestrator", "run_as_backend", "run_as_frontend", "start_app"]
