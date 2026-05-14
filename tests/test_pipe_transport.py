import time

import numpy as np
import pytest

from compneurovis.core import Field, LayoutSpec, AppSpec
from compneurovis.core import ActorBase, ActorRole
from compneurovis.backends import BackendBase
from compneurovis.frontends import FrontendBase
from compneurovis.messages import (
    EntityClicked,
    Error,
    FieldReplace,
    SetControl,
    StatePatch,
    Status,
    command_message,
    update_message,
)
from compneurovis.transports import PipeTransport
from compneurovis.transports.pipe import _sleep_to_backend_cadence


def _dummy_app_spec() -> AppSpec:
    field = Field(
        id="demo",
        values=np.array([1.0, 2.0], dtype=np.float32),
        dims=("x",),
        coords={"x": np.array([0.0, 1.0], dtype=np.float32)},
    )
    return AppSpec(fields={"demo": field}, geometries={}, views={}, layout=LayoutSpec(title="Dummy"))


class DummyBackend(BackendBase):
    def initialize(self, app_spec: AppSpec) -> None:
        pass

    def is_live(self) -> bool:
        return False

    def advance(self) -> None:
        return None

    def handle(self, message) -> None:
        command = message.payload
        if isinstance(command, SetControl):
            self.emit_update(
                FieldReplace(
                    field_id="demo",
                    values=np.array([command.value, command.value + 1], dtype=np.float32),
                )
            )


class DummyInteractionBackend(BackendBase):
    def initialize(self, app_spec: AppSpec) -> None:
        pass

    def is_live(self) -> bool:
        return False

    def advance(self) -> None:
        return None

    def handle(self, message) -> None:
        command = message.payload
        if isinstance(command, EntityClicked):
            self.emit_update(StatePatch({"selected_trace_entity_ids": [command.entity_id]}))
            self.emit_update(Status(f"Clicked {command.entity_id}", 1000))


class FailingInitializeBackend(BackendBase):
    def initialize(self, app_spec: AppSpec) -> None:
        raise RuntimeError("intentional backend init failure")

    def is_live(self) -> bool:
        return False

    def advance(self) -> None:
        return None

    def handle(self, message) -> None:
        command = message.payload
        del command


class FastLiveBackend(BackendBase):
    def __init__(self):
        super().__init__()
        self.tick_count = 0

    def initialize(self, app_spec: AppSpec) -> None:
        pass

    def advance(self) -> None:
        self.tick_count += 1
        self.emit_update(Status(f"tick:{self.tick_count}", 0))

    def handle(self, message) -> None:
        command = message.payload
        del command

    def idle_sleep(self) -> float:
        return 0.05


class FloodLiveBackend(BackendBase):
    def __init__(self):
        super().__init__()
        self.tick_count = 0

    def initialize(self, app_spec: AppSpec) -> None:
        pass

    def advance(self) -> None:
        self.tick_count += 1
        self.emit_update(Status(f"tick:{self.tick_count}", 0))

    def handle(self, message) -> None:
        command = message.payload
        del command

    def idle_sleep(self) -> float:
        return 0.0


def make_dummy_backend() -> DummyBackend:
    return DummyBackend()


def test_backend_and_frontend_share_actor_contract():
    class DummyFrontend(FrontendBase):
        def handle(self, message) -> None:
            del message

    backend = DummyBackend()
    frontend = DummyFrontend()

    assert isinstance(backend, ActorBase)
    assert isinstance(frontend, ActorBase)
    assert backend.role is ActorRole.BACKEND
    assert frontend.role is ActorRole.FRONTEND

    backend.emit(update_message(Status("ready", 0)))
    frontend.emit_command(SetControl("demo", 3.0))

    assert backend.take_outbound_messages()[0].payload == Status("ready", 0)
    assert frontend.take_outbound_messages()[0].payload == SetControl("demo", 3.0)


def update_payloads(messages):
    return [message.payload for message in messages]


def test_pipe_transport_roundtrip():
    transport = PipeTransport(DummyBackend, provided_app_spec=_dummy_app_spec())
    transport.start()
    try:
        transport.send(command_message(SetControl("demo", 5.0)))
        deadline = time.time() + 5
        field_replace = None
        while time.time() < deadline and field_replace is None:
            for update in update_payloads(transport.poll()):
                if isinstance(update, FieldReplace):
                    field_replace = update
                    break
            time.sleep(0.05)

        assert field_replace is not None
        assert np.allclose(field_replace.values, np.array([5.0, 6.0], dtype=np.float32))
    finally:
        transport.stop()


def test_pipe_transport_requires_lazy_session_source():
    with pytest.raises(TypeError, match="PipeTransport requires a Backend subclass or top-level zero-argument factory"):
        PipeTransport(DummyBackend())


def test_pipe_transport_roundtrip_from_factory():
    transport = PipeTransport(make_dummy_backend, provided_app_spec=_dummy_app_spec())
    transport.start()
    try:
        transport.send(command_message(SetControl("demo", 7.0)))
        deadline = time.time() + 5
        field_replace = None
        while time.time() < deadline and field_replace is None:
            for update in update_payloads(transport.poll()):
                if isinstance(update, FieldReplace):
                    field_replace = update
                    break
            time.sleep(0.05)

        assert field_replace is not None
        assert np.allclose(field_replace.values, np.array([7.0, 8.0], dtype=np.float32))
    finally:
        transport.stop()


def test_pipe_transport_roundtrip_interaction_commands():
    transport = PipeTransport(DummyInteractionBackend, provided_app_spec=_dummy_app_spec())
    transport.start()
    try:
        transport.send(command_message(EntityClicked("seg-a")))
        deadline = time.time() + 5
        state_patch = None
        status = None
        while time.time() < deadline and (state_patch is None or status is None):
            for update in update_payloads(transport.poll()):
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
    transport = PipeTransport(FailingInitializeBackend, provided_app_spec=AppSpec())
    transport.start()
    try:
        deadline = time.time() + 5
        error = None
        while time.time() < deadline and error is None:
            for update in update_payloads(transport.poll()):
                if isinstance(update, Error):
                    error = update
                    break
            time.sleep(0.05)

        assert error is not None
        assert "RuntimeError" in error.message
        assert "intentional backend init failure" in error.message
    finally:
        transport.stop()


def test_sleep_to_backend_cadence_waits_out_remaining_tick_time():
    backend = FastLiveBackend()
    started = time.monotonic()
    _sleep_to_backend_cadence(backend, started)
    elapsed = time.monotonic() - started
    assert elapsed >= 0.04


def _wait_for_first_update(transport, *, deadline_s: float = 8.0) -> list:
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        updates = update_payloads(transport.poll())
        if updates:
            return updates
        time.sleep(0.05)
    return []


def test_live_pipe_transport_respects_idle_sleep_cadence():
    transport = PipeTransport(FastLiveBackend, provided_app_spec=AppSpec())
    transport.start()
    try:
        # Subprocess startup on Windows (spawn) can take several seconds.
        # Wait until the first update arrives, then measure pacing over 0.25s.
        first = _wait_for_first_update(transport)
        assert first, "expected backend to start within 8 seconds"
        time.sleep(0.25)
        updates = update_payloads(transport.poll())
        all_updates = first + updates
        statuses = [update for update in all_updates if isinstance(update, Status)]
        assert statuses, "expected paced live status updates"
        last_tick = max(int(status.message.split(":")[1]) for status in statuses)
        assert last_tick < 20
    finally:
        transport.stop()


def test_pipe_transport_bounds_update_drain_per_poll_to_preserve_ui_responsiveness():
    transport = PipeTransport(FloodLiveBackend, provided_app_spec=AppSpec())
    transport._max_update_payloads_per_poll = 8
    transport._max_poll_duration_s = 1.0
    transport.start()
    try:
        # Wait for subprocess startup, then let the flood backend accumulate updates.
        first = _wait_for_first_update(transport)
        assert first, "expected backend to start within 8 seconds"
        time.sleep(0.2)
        updates = update_payloads(transport.poll())
        statuses = [update for update in updates if isinstance(update, Status)]
        assert statuses, "expected flood backend updates"
        assert len(updates) <= 8
        assert transport._last_poll_payload_count <= 8
        assert transport._last_poll_truncated is True
        assert transport._last_poll_more_pending is True
    finally:
        transport.stop()
