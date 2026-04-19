from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
import traceback
from multiprocessing import Pipe, Process

from PyQt6 import QtCore

from compneurovis._perf import clear_perf_logging_configuration, configure_perf_logging, perf_log
from compneurovis.core.scene import DiagnosticsSpec
from compneurovis.session.base import Session, SessionSource, resolve_session_source
from compneurovis.session.protocol import SceneReady, Error, SessionCommand, SessionUpdate, StopSession

DEFAULT_MAX_UPDATE_PAYLOADS_PER_POLL = 16
DEFAULT_MAX_POLL_DURATION_S = 0.004


def _update_type_counts(updates: list[SessionUpdate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for update in updates:
        name = type(update).__name__
        counts[name] = counts.get(name, 0) + 1
    return counts


def _update_sample_counts(updates: list[SessionUpdate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for update in updates:
        coord_values = getattr(update, "coord_values", None)
        field_id = getattr(update, "field_id", None)
        if coord_values is None or field_id is None:
            continue
        counts[str(field_id)] = int(getattr(coord_values, "shape", [len(coord_values)])[0])
    return counts


def _sleep_to_session_cadence(session: Session, started_at: float) -> None:
    delay = float(session.idle_sleep())
    if delay <= 0:
        return
    remaining = delay - (time.monotonic() - started_at)
    if remaining > 0:
        time.sleep(remaining)


def _apply_perf_logging_configuration(diagnostics: DiagnosticsSpec | None) -> None:
    if diagnostics is None:
        clear_perf_logging_configuration()
    else:
        configure_perf_logging(diagnostics)


def _session_process(session_source: SessionSource, diagnostics: DiagnosticsSpec | None, update_pipe, command_pipe) -> None:
    session: Session | None = None
    try:
        _apply_perf_logging_configuration(diagnostics)
        session = resolve_session_source(session_source)
        scene = session.initialize()
        perf_log(
            "session_worker",
            "initialize",
            session_type=type(session).__name__,
            has_scene=scene is not None,
            is_live=session.is_live(),
            idle_sleep_s=session.idle_sleep(),
        )
        if scene is not None:
            update_pipe.send(SceneReady(scene))

        while True:
            tick_started = time.monotonic()
            while command_pipe.poll():
                command = command_pipe.recv()
                if isinstance(command, StopSession):
                    return
                command_started = time.monotonic()
                session.handle(command)
                perf_log(
                    "session_worker",
                    "handle_command",
                    command_type=type(command).__name__,
                    control_id=getattr(command, "control_id", None),
                    action_id=getattr(command, "action_id", None),
                    key=getattr(command, "key", None),
                    entity_id=getattr(command, "entity_id", None),
                    value=getattr(command, "value", None),
                    duration_ms=round((time.monotonic() - command_started) * 1000.0, 3),
                )

            advance_started = time.monotonic()
            if session.is_live():
                session.advance()
            advance_ms = round((time.monotonic() - advance_started) * 1000.0, 3)

            updates = session.read_updates()
            if updates:
                update_pipe.send(updates if len(updates) > 1 else updates[0])
            perf_log(
                "session_worker",
                "tick",
                session_type=type(session).__name__,
                is_live=session.is_live(),
                advance_ms=advance_ms,
                emitted_update_count=len(updates),
                emitted_update_types=_update_type_counts(updates),
                emitted_field_append_samples=_update_sample_counts(updates),
            )
            _sleep_to_session_cadence(session, tick_started)
    except Exception as exc:  # pragma: no cover - safety net for worker errors
        detail = "".join(traceback.format_exception(exc))
        perf_log("session_worker", "error", error_type=type(exc).__name__, message=str(exc))
        update_pipe.send(Error(detail))
    finally:
        try:
            if session is not None:
                session.shutdown()
        finally:
            update_pipe.close()
            command_pipe.close()


def _session_process_queue(
    session_source: SessionSource,
    diagnostics: DiagnosticsSpec | None,
    update_queue,
    command_queue,
    stop_event=None,
) -> None:
    session: Session | None = None
    try:
        _apply_perf_logging_configuration(diagnostics)
        session = resolve_session_source(session_source)
        scene = session.initialize()
        perf_log(
            "session_worker",
            "initialize",
            session_type=type(session).__name__,
            has_scene=scene is not None,
            is_live=session.is_live(),
            idle_sleep_s=session.idle_sleep(),
            transport_mode="thread",
        )
        if scene is not None:
            update_queue.put(SceneReady(scene))

        while True:
            tick_started = time.monotonic()
            if stop_event is not None and stop_event.is_set():
                return
            try:
                command = command_queue.get_nowait()
            except queue.Empty:
                command = None
            if command is not None:
                if isinstance(command, StopSession):
                    return
                command_started = time.monotonic()
                session.handle(command)
                perf_log(
                    "session_worker",
                    "handle_command",
                    transport_mode="thread",
                    command_type=type(command).__name__,
                    control_id=getattr(command, "control_id", None),
                    action_id=getattr(command, "action_id", None),
                    key=getattr(command, "key", None),
                    entity_id=getattr(command, "entity_id", None),
                    value=getattr(command, "value", None),
                    duration_ms=round((time.monotonic() - command_started) * 1000.0, 3),
                )

            advance_started = time.monotonic()
            if session.is_live():
                session.advance()
            advance_ms = round((time.monotonic() - advance_started) * 1000.0, 3)

            updates = session.read_updates()
            if updates:
                update_queue.put(updates if len(updates) > 1 else updates[0])
            perf_log(
                "session_worker",
                "tick",
                session_type=type(session).__name__,
                is_live=session.is_live(),
                transport_mode="thread",
                advance_ms=advance_ms,
                emitted_update_count=len(updates),
                emitted_update_types=_update_type_counts(updates),
                emitted_field_append_samples=_update_sample_counts(updates),
            )
            _sleep_to_session_cadence(session, tick_started)
    except Exception as exc:  # pragma: no cover - safety net for worker errors
        detail = "".join(traceback.format_exception(exc))
        perf_log("session_worker", "error", transport_mode="thread", error_type=type(exc).__name__, message=str(exc))
        update_queue.put(Error(detail))
    finally:
        if session is not None:
            session.shutdown()


class PipeTransport(QtCore.QObject):
    def __init__(self, session: SessionSource, diagnostics: DiagnosticsSpec | None = None, parent=None) -> None:
        super().__init__(parent)
        if isinstance(session, Session):
            raise TypeError(
                "PipeTransport requires a Session subclass or top-level zero-argument factory. "
                "Do not pass an already-created session instance."
            )
        self.session = session
        self.diagnostics = diagnostics
        self._mode = "pipe"
        self._dead = False
        self._children = ()
        self.thread = None
        self._stop_event = None
        self._max_update_payloads_per_poll = DEFAULT_MAX_UPDATE_PAYLOADS_PER_POLL
        self._max_poll_duration_s = DEFAULT_MAX_POLL_DURATION_S
        self._last_poll_payload_count = 0
        self._last_poll_truncated = False
        self._last_poll_more_pending = False
        self._last_poll_duration_ms = 0.0
        try:
            self.update_parent, update_child = Pipe()
            self.command_child, command_parent = Pipe()
            self.command_parent = command_parent
            self.process = Process(
                target=_session_process,
                args=(session, diagnostics, update_child, self.command_child),
            )
            self._children = (update_child, self.command_child)
        except PermissionError:
            self._mode = "thread"
            self.process = None
            self.update_parent = queue.Queue()
            self.command_parent = queue.Queue()
            self._stop_event = threading.Event()
            self.thread = threading.Thread(
                target=_session_process_queue,
                args=(session, diagnostics, self.update_parent, self.command_parent, self._stop_event),
                daemon=True,
            )

    def start(self) -> None:
        if self._mode == "thread":
            self.thread.start()
        else:
            self.process.start()
            for child in self._children:
                child.close()

    def poll_updates(self) -> list[SessionUpdate]:
        poll_started = time.monotonic()
        updates: list[SessionUpdate] = []
        payload_count = 0
        truncated = False
        more_pending = False

        def append_payload(payload) -> None:
            if isinstance(payload, list):
                updates.extend(payload)
            else:
                updates.append(payload)

        if self._mode == "pipe":
            try:
                while not self._dead:
                    if payload_count >= self._max_update_payloads_per_poll or time.monotonic() - poll_started >= self._max_poll_duration_s:
                        truncated = True
                        more_pending = self.update_parent.poll()
                        break
                    if not self.update_parent.poll():
                        break
                    append_payload(self.update_parent.recv())
                    payload_count += 1
            except (BrokenPipeError, EOFError, OSError) as exc:
                self._dead = True
                updates.append(Error(f"Worker process ended unexpectedly: {exc}"))
        else:
            while True:
                if payload_count >= self._max_update_payloads_per_poll or time.monotonic() - poll_started >= self._max_poll_duration_s:
                    truncated = True
                    more_pending = not self.update_parent.empty()
                    break
                try:
                    append_payload(self.update_parent.get_nowait())
                    payload_count += 1
                except queue.Empty:
                    break
        duration_ms = round((time.monotonic() - poll_started) * 1000.0, 3)
        self._last_poll_payload_count = payload_count
        self._last_poll_truncated = truncated
        self._last_poll_more_pending = more_pending
        self._last_poll_duration_ms = duration_ms
        perf_log(
            "transport",
            "poll_updates",
            mode=self._mode,
            payload_count=payload_count,
            update_count=len(updates),
            update_types=_update_type_counts(updates),
            field_append_samples=_update_sample_counts(updates),
            truncated=truncated,
            more_pending=more_pending,
            duration_ms=duration_ms,
        )
        return updates

    def send_command(self, command: SessionCommand) -> None:
        perf_log(
            "transport",
            "send_command",
            mode=self._mode,
            command_type=type(command).__name__,
            control_id=getattr(command, "control_id", None),
            action_id=getattr(command, "action_id", None),
            key=getattr(command, "key", None),
            entity_id=getattr(command, "entity_id", None),
            value=getattr(command, "value", None),
        )
        if self._mode == "pipe":
            self.command_parent.send(command)
        else:
            self.command_parent.put(command)

    def stop(self) -> None:
        if self._mode == "thread":
            if self.thread is not None and self.thread.is_alive():
                self._stop_event.set()
                self.command_parent.put(StopSession())
                self.thread.join(timeout=1)
            return

        if self.process.is_alive():
            try:
                self.send_command(StopSession())
                self.process.join(1)
            except Exception:
                pass
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
        self.update_parent.close()
        self.command_parent.close()


def configure_multiprocessing() -> None:
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
