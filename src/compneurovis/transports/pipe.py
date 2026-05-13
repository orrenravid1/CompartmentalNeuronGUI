from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
import traceback
from multiprocessing import Pipe, Process

from PyQt6 import QtCore

from compneurovis._perf import clear_perf_logging_configuration, configure_perf_logging, perf_log
from compneurovis.core.app import DiagnosticsSpec
from compneurovis.backends.base import Backend, BackendSource, resolve_backend_source
from compneurovis.messages import (
    AppSpecReady,
    CommandMessage,
    Error,
    StopBackend,
    UpdateMessage,
    command_message,
    update_message,
)

DEFAULT_MAX_UPDATE_PAYLOADS_PER_POLL = 16
DEFAULT_MAX_POLL_DURATION_S = 0.004


def _update_type_counts(messages: list[UpdateMessage]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for message in messages:
        update = message.payload
        name = type(update).__name__
        counts[name] = counts.get(name, 0) + 1
    return counts


def _update_sample_counts(messages: list[UpdateMessage]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for message in messages:
        update = message.payload
        coord_values = getattr(update, "coord_values", None)
        field_id = getattr(update, "field_id", None)
        if coord_values is None or field_id is None:
            continue
        counts[str(field_id)] = int(getattr(coord_values, "shape", [len(coord_values)])[0])
    return counts


def _sleep_to_backend_cadence(backend: Backend, started_at: float) -> None:
    delay = float(backend.idle_sleep())
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


def _backend_process(backend_source: BackendSource, diagnostics: DiagnosticsSpec | None, update_pipe, command_pipe) -> None:
    backend: Backend | None = None
    try:
        _apply_perf_logging_configuration(diagnostics)
        backend = resolve_backend_source(backend_source)
        app_spec = backend.initialize()
        perf_log(
            "backend_worker",
            "initialize",
            backend_type=type(backend).__name__,
            has_app_spec=app_spec is not None,
            is_live=backend.is_live(),
            idle_sleep_s=backend.idle_sleep(),
        )
        if app_spec is not None:
            update_pipe.send(update_message(AppSpecReady(app_spec)))

        while True:
            tick_started = time.monotonic()
            while command_pipe.poll():
                message = command_pipe.recv()
                command = message.payload
                if isinstance(command, StopBackend):
                    return
                command_started = time.monotonic()
                backend.handle(message)
                perf_log(
                    "backend_worker",
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
            if backend.is_live():
                backend.advance()
            advance_ms = round((time.monotonic() - advance_started) * 1000.0, 3)

            messages = backend.take_outbound_messages()
            if messages:
                update_pipe.send(messages if len(messages) > 1 else messages[0])
            perf_log(
                "backend_worker",
                "tick",
                backend_type=type(backend).__name__,
                is_live=backend.is_live(),
                advance_ms=advance_ms,
                emitted_update_count=len(messages),
                emitted_update_types=_update_type_counts(messages),
                emitted_field_append_samples=_update_sample_counts(messages),
            )
            _sleep_to_backend_cadence(backend, tick_started)
    except Exception as exc:  # pragma: no cover - safety net for worker errors
        detail = "".join(traceback.format_exception(exc))
        perf_log("backend_worker", "error", error_type=type(exc).__name__, message=str(exc))
        update_pipe.send(update_message(Error(detail)))
    finally:
        try:
            if backend is not None:
                backend.shutdown()
        finally:
            update_pipe.close()
            command_pipe.close()


def _backend_process_queue(
    backend_source: BackendSource,
    diagnostics: DiagnosticsSpec | None,
    update_queue,
    command_queue,
    stop_event=None,
) -> None:
    backend: Backend | None = None
    try:
        _apply_perf_logging_configuration(diagnostics)
        backend = resolve_backend_source(backend_source)
        app_spec = backend.initialize()
        perf_log(
            "backend_worker",
            "initialize",
            backend_type=type(backend).__name__,
            has_app_spec=app_spec is not None,
            is_live=backend.is_live(),
            idle_sleep_s=backend.idle_sleep(),
            transport_mode="thread",
        )
        if app_spec is not None:
            update_queue.put(update_message(AppSpecReady(app_spec)))

        while True:
            tick_started = time.monotonic()
            if stop_event is not None and stop_event.is_set():
                return
            try:
                message = command_queue.get_nowait()
            except queue.Empty:
                message = None
            if message is not None:
                command = message.payload
                if isinstance(command, StopBackend):
                    return
                command_started = time.monotonic()
                backend.handle(message)
                perf_log(
                    "backend_worker",
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
            if backend.is_live():
                backend.advance()
            advance_ms = round((time.monotonic() - advance_started) * 1000.0, 3)

            messages = backend.take_outbound_messages()
            if messages:
                update_queue.put(messages if len(messages) > 1 else messages[0])
            perf_log(
                "backend_worker",
                "tick",
                backend_type=type(backend).__name__,
                is_live=backend.is_live(),
                transport_mode="thread",
                advance_ms=advance_ms,
                emitted_update_count=len(messages),
                emitted_update_types=_update_type_counts(messages),
                emitted_field_append_samples=_update_sample_counts(messages),
            )
            _sleep_to_backend_cadence(backend, tick_started)
    except Exception as exc:  # pragma: no cover - safety net for worker errors
        detail = "".join(traceback.format_exception(exc))
        perf_log("backend_worker", "error", transport_mode="thread", error_type=type(exc).__name__, message=str(exc))
        update_queue.put(update_message(Error(detail)))
    finally:
        if backend is not None:
            backend.shutdown()


class PipeTransport(QtCore.QObject):
    def __init__(self, backend: BackendSource, diagnostics: DiagnosticsSpec | None = None, parent=None) -> None:
        super().__init__(parent)
        if isinstance(backend, Backend):
            raise TypeError(
                "PipeTransport requires a Backend subclass or top-level zero-argument factory. "
                "Do not pass an already-created backend instance."
            )
        self.backend = backend
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
                target=_backend_process,
                args=(backend, diagnostics, update_child, self.command_child),
            )
            self._children = (update_child, self.command_child)
        except PermissionError:
            self._mode = "thread"
            self.process = None
            self.update_parent = queue.Queue()
            self.command_parent = queue.Queue()
            self._stop_event = threading.Event()
            self.thread = threading.Thread(
                target=_backend_process_queue,
                args=(backend, diagnostics, self.update_parent, self.command_parent, self._stop_event),
                daemon=True,
            )

    def start(self) -> None:
        if self._mode == "thread":
            self.thread.start()
        else:
            self.process.start()
            for child in self._children:
                child.close()

    def poll(self) -> list[UpdateMessage]:
        poll_started = time.monotonic()
        messages: list[UpdateMessage] = []
        payload_count = 0
        truncated = False
        more_pending = False

        def append_payload(payload) -> None:
            if isinstance(payload, list):
                messages.extend(payload)
            else:
                messages.append(payload)

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
                messages.append(update_message(Error(f"Worker process ended unexpectedly: {exc}")))
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
            "poll",
            mode=self._mode,
            payload_count=payload_count,
            update_count=len(messages),
            update_types=_update_type_counts(messages),
            field_append_samples=_update_sample_counts(messages),
            truncated=truncated,
            more_pending=more_pending,
            duration_ms=duration_ms,
        )
        return messages

    def send(self, message: CommandMessage) -> None:
        command = message.payload
        perf_log(
            "transport",
            "send",
            mode=self._mode,
            command_type=type(command).__name__,
            control_id=getattr(command, "control_id", None),
            action_id=getattr(command, "action_id", None),
            key=getattr(command, "key", None),
            entity_id=getattr(command, "entity_id", None),
            value=getattr(command, "value", None),
        )
        if self._mode == "pipe":
            self.command_parent.send(message)
        else:
            self.command_parent.put(message)

    def stop(self) -> None:
        if self._mode == "thread":
            if self.thread is not None and self.thread.is_alive():
                self._stop_event.set()
                self.command_parent.put(command_message(StopBackend()))
                self.thread.join(timeout=1)
            return

        if self.process.is_alive():
            try:
                self.send(command_message(StopBackend()))
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
