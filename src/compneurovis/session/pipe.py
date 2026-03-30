from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
import traceback
from multiprocessing import Pipe, Process

from PyQt6 import QtCore

from compneurovis.session.base import Session
from compneurovis.session.protocol import DocumentReady, Error, SessionCommand, SessionUpdate, StopSession


def _session_process(session: Session, update_pipe, command_pipe) -> None:
    try:
        document = session.initialize()
        if document is not None:
            update_pipe.send(DocumentReady(document))

        while True:
            while command_pipe.poll():
                command = command_pipe.recv()
                if isinstance(command, StopSession):
                    return
                session.handle(command)

            if session.is_live():
                session.advance()
            else:
                time.sleep(session.idle_sleep())

            for update in session.read_updates():
                update_pipe.send(update)
    except Exception as exc:  # pragma: no cover - safety net for worker errors
        detail = "".join(traceback.format_exception(exc))
        update_pipe.send(Error(detail))
    finally:
        try:
            session.shutdown()
        finally:
            update_pipe.close()
            command_pipe.close()


def _session_process_queue(session: Session, update_queue, command_queue, stop_event=None) -> None:
    try:
        document = session.initialize()
        if document is not None:
            update_queue.put(DocumentReady(document))

        while True:
            if stop_event is not None and stop_event.is_set():
                return
            try:
                command = command_queue.get_nowait()
            except queue.Empty:
                command = None
            if command is not None:
                if isinstance(command, StopSession):
                    return
                session.handle(command)

            if session.is_live():
                session.advance()
            else:
                time.sleep(session.idle_sleep())

            for update in session.read_updates():
                update_queue.put(update)
    except Exception as exc:  # pragma: no cover - safety net for worker errors
        detail = "".join(traceback.format_exception(exc))
        update_queue.put(Error(detail))
    finally:
        session.shutdown()


class PipeTransport(QtCore.QObject):
    def __init__(self, session: Session, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self._mode = "pipe"
        self._dead = False
        self._children = ()
        self.thread = None
        self._stop_event = None
        try:
            self.update_parent, update_child = Pipe()
            self.command_child, command_parent = Pipe()
            self.command_parent = command_parent
            self.process = Process(
                target=_session_process,
                args=(session, update_child, self.command_child),
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
                args=(session, self.update_parent, self.command_parent, self._stop_event),
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
        updates: list[SessionUpdate] = []
        if self._mode == "pipe":
            if self._dead:
                return updates
            try:
                while self.update_parent.poll():
                    updates.append(self.update_parent.recv())
            except (BrokenPipeError, EOFError, OSError) as exc:
                self._dead = True
                updates.append(Error(f"Worker process ended unexpectedly: {exc}"))
        else:
            while True:
                try:
                    updates.append(self.update_parent.get_nowait())
                except queue.Empty:
                    break
        return updates

    def send_command(self, command: SessionCommand) -> None:
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
