import time

import numpy as np
import pytest

from compneurovis.core import Field, LayoutSpec, Scene
from compneurovis.session import BufferedSession, EntityClicked, Error, FieldReplace, PipeTransport, SetControl, StatePatch, Status


class DummySession(BufferedSession):
    def initialize(self):
        field = Field(
            id="demo",
            values=np.array([1.0, 2.0], dtype=np.float32),
            dims=("x",),
            coords={"x": np.array([0.0, 1.0], dtype=np.float32)},
        )
        return Scene(fields={"demo": field}, geometries={}, views={}, layout=LayoutSpec(title="Dummy"))

    def is_live(self) -> bool:
        return False

    def advance(self) -> None:
        return None

    def handle(self, command) -> None:
        if isinstance(command, SetControl):
            self.emit(
                FieldReplace(
                    field_id="demo",
                    values=np.array([command.value, command.value + 1], dtype=np.float32),
                )
            )


class DummyInteractionSession(BufferedSession):
    def initialize(self):
        field = Field(
            id="demo",
            values=np.array([1.0], dtype=np.float32),
            dims=("x",),
            coords={"x": np.array([0.0], dtype=np.float32)},
        )
        return Scene(fields={"demo": field}, geometries={}, views={}, layout=LayoutSpec(title="Dummy"))

    def is_live(self) -> bool:
        return False

    def advance(self) -> None:
        return None

    def handle(self, command) -> None:
        if isinstance(command, EntityClicked):
            self.emit(StatePatch({"selected_trace_entity_ids": [command.entity_id]}))
            self.emit(Status(f"Clicked {command.entity_id}", 1000))


class FailingInitializeSession(BufferedSession):
    def initialize(self):
        raise RuntimeError("intentional session init failure")

    def is_live(self) -> bool:
        return False

    def advance(self) -> None:
        return None

    def handle(self, command) -> None:
        del command


def make_dummy_session() -> DummySession:
    return DummySession()


def test_pipe_transport_roundtrip():
    transport = PipeTransport(DummySession)
    transport.start()
    try:
        deadline = time.time() + 5
        updates = []
        while time.time() < deadline and not updates:
            updates = transport.poll_updates()
            time.sleep(0.05)
        assert updates, "expected initial scene update"

        transport.send_command(SetControl("demo", 5.0))
        deadline = time.time() + 5
        field_replace = None
        while time.time() < deadline and field_replace is None:
            for update in transport.poll_updates():
                if isinstance(update, FieldReplace):
                    field_replace = update
                    break
            time.sleep(0.05)

        assert field_replace is not None
        assert np.allclose(field_replace.values, np.array([5.0, 6.0], dtype=np.float32))
    finally:
        transport.stop()


def test_pipe_transport_requires_lazy_session_source():
    with pytest.raises(TypeError, match="PipeTransport requires a Session subclass or top-level zero-argument factory"):
        PipeTransport(DummySession())


def test_pipe_transport_roundtrip_from_factory():
    transport = PipeTransport(make_dummy_session)
    transport.start()
    try:
        deadline = time.time() + 5
        updates = []
        while time.time() < deadline and not updates:
            updates = transport.poll_updates()
            time.sleep(0.05)
        assert updates, "expected initial scene update"

        transport.send_command(SetControl("demo", 7.0))
        deadline = time.time() + 5
        field_replace = None
        while time.time() < deadline and field_replace is None:
            for update in transport.poll_updates():
                if isinstance(update, FieldReplace):
                    field_replace = update
                    break
            time.sleep(0.05)

        assert field_replace is not None
        assert np.allclose(field_replace.values, np.array([7.0, 8.0], dtype=np.float32))
    finally:
        transport.stop()


def test_pipe_transport_roundtrip_interaction_commands():
    transport = PipeTransport(DummyInteractionSession)
    transport.start()
    try:
        deadline = time.time() + 5
        updates = []
        while time.time() < deadline and not updates:
            updates = transport.poll_updates()
            time.sleep(0.05)
        assert updates, "expected initial scene update"

        transport.send_command(EntityClicked("seg-a"))
        deadline = time.time() + 5
        state_patch = None
        status = None
        while time.time() < deadline and (state_patch is None or status is None):
            for update in transport.poll_updates():
                if isinstance(update, StatePatch):
                    state_patch = update
                elif isinstance(update, Status):
                    status = update
            time.sleep(0.05)

        assert state_patch is not None
        assert state_patch.updates == {"selected_trace_entity_ids": ["seg-a"]}
        assert status is not None
        assert status.message == "Clicked seg-a"
        assert status.timeout_ms == 1000
    finally:
        transport.stop()


def test_pipe_transport_surfaces_worker_initialize_errors():
    transport = PipeTransport(FailingInitializeSession)
    transport.start()
    try:
        deadline = time.time() + 5
        error = None
        while time.time() < deadline and error is None:
            for update in transport.poll_updates():
                if isinstance(update, Error):
                    error = update
                    break
            time.sleep(0.05)

        assert error is not None
        assert "RuntimeError" in error.message
        assert "intentional session init failure" in error.message
    finally:
        transport.stop()
