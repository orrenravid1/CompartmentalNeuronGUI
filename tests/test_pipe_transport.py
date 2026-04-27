import time

import numpy as np
import pytest

from compneurovis.core import Field, LayoutSpec, Scene
from compneurovis.session import BufferedSession, EntityClicked, Error, FieldReplace, PipeTransport, SetControl, StatePatch, Status
from compneurovis.session.pipe import _sleep_to_session_cadence


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


class FastLiveSession(BufferedSession):
    def __init__(self):
        super().__init__()
        self.tick_count = 0

    def initialize(self):
        return None

    def advance(self) -> None:
        self.tick_count += 1
        self.emit(Status(f"tick:{self.tick_count}", 0))

    def handle(self, command) -> None:
        del command

    def idle_sleep(self) -> float:
        return 0.05


class FloodLiveSession(BufferedSession):
    def __init__(self):
        super().__init__()
        self.tick_count = 0

    def initialize(self):
        return None

    def advance(self) -> None:
        self.tick_count += 1
        self.emit(Status(f"tick:{self.tick_count}", 0))

    def handle(self, command) -> None:
        del command

    def idle_sleep(self) -> float:
        return 0.0


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


def test_sleep_to_session_cadence_waits_out_remaining_tick_time():
    session = FastLiveSession()
    started = time.monotonic()
    _sleep_to_session_cadence(session, started)
    elapsed = time.monotonic() - started
    assert elapsed >= 0.04


def _wait_for_first_update(transport, *, deadline_s: float = 8.0) -> list:
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        updates = transport.poll_updates()
        if updates:
            return updates
        time.sleep(0.05)
    return []


def test_live_pipe_transport_respects_idle_sleep_cadence():
    transport = PipeTransport(FastLiveSession)
    transport.start()
    try:
        # Subprocess startup on Windows (spawn) can take several seconds.
        # Wait until the first update arrives, then measure pacing over 0.25s.
        first = _wait_for_first_update(transport)
        assert first, "expected session to start within 8 seconds"
        time.sleep(0.25)
        updates = transport.poll_updates()
        all_updates = first + updates
        statuses = [update for update in all_updates if isinstance(update, Status)]
        assert statuses, "expected paced live status updates"
        last_tick = max(int(status.message.split(":")[1]) for status in statuses)
        assert last_tick < 20
    finally:
        transport.stop()


def test_pipe_transport_bounds_update_drain_per_poll_to_preserve_ui_responsiveness():
    transport = PipeTransport(FloodLiveSession)
    transport._max_update_payloads_per_poll = 8
    transport._max_poll_duration_s = 1.0
    transport.start()
    try:
        # Wait for subprocess startup, then let the flood session accumulate updates.
        first = _wait_for_first_update(transport)
        assert first, "expected session to start within 8 seconds"
        time.sleep(0.2)
        updates = transport.poll_updates()
        statuses = [update for update in updates if isinstance(update, Status)]
        assert statuses, "expected flood session updates"
        assert len(updates) <= 8
        assert transport._last_poll_payload_count <= 8
        assert transport._last_poll_truncated is True
        assert transport._last_poll_more_pending is True
    finally:
        transport.stop()
