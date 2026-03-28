import time

import numpy as np

from compneurovis.core import Document, Field, LayoutSpec
from compneurovis.session import BufferedSession, FieldUpdate, PipeTransport, SetControl


class DummySession(BufferedSession):
    def initialize(self):
        field = Field(
            id="demo",
            values=np.array([1.0, 2.0], dtype=np.float32),
            dims=("x",),
            coords={"x": np.array([0.0, 1.0], dtype=np.float32)},
        )
        return Document(fields={"demo": field}, geometries={}, views={}, layout=LayoutSpec(title="Dummy"))

    def is_live(self) -> bool:
        return False

    def advance(self) -> None:
        return None

    def handle(self, command) -> None:
        if isinstance(command, SetControl):
            self.emit(
                FieldUpdate(
                    field_id="demo",
                    values=np.array([command.value, command.value + 1], dtype=np.float32),
                )
            )


def test_pipe_transport_roundtrip():
    transport = PipeTransport(DummySession())
    transport.start()
    try:
        deadline = time.time() + 5
        updates = []
        while time.time() < deadline and not updates:
            updates = transport.poll_updates()
            time.sleep(0.05)
        assert updates, "expected initial document update"

        transport.send_command(SetControl("demo", 5.0))
        deadline = time.time() + 5
        field_update = None
        while time.time() < deadline and field_update is None:
            for update in transport.poll_updates():
                if isinstance(update, FieldUpdate):
                    field_update = update
                    break
            time.sleep(0.05)

        assert field_update is not None
        assert np.allclose(field_update.values, np.array([5.0, 6.0], dtype=np.float32))
    finally:
        transport.stop()

