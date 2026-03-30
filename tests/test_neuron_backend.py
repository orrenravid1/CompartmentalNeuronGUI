import numpy as np

from compneurovis.backends.neuron.document import NeuronDocumentBuilder
from compneurovis.backends.neuron.session import NeuronSession
from compneurovis.core import MorphologyGeometry
from compneurovis.session import EntityClicked, FieldReplace


def _geometry() -> MorphologyGeometry:
    return MorphologyGeometry(
        id="morphology",
        positions=np.zeros((2, 3), dtype=np.float32),
        orientations=np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0),
        radii=np.ones(2, dtype=np.float32),
        lengths=np.ones(2, dtype=np.float32),
        entity_ids=("seg-a", "seg-b"),
        section_names=("sec-a", "sec-b"),
        xlocs=np.array([0.1, 0.9], dtype=np.float32),
        labels=("seg-a", "seg-b"),
    )


def test_neuron_document_builder_splits_display_and_trace_fields():
    document = NeuronDocumentBuilder.build_document(
        geometry=_geometry(),
        display_values=np.array([1.0, 2.0], dtype=np.float32),
        trace_values=np.array([[1.0, 2.0]], dtype=np.float32),
        trace_segment_ids=np.array(["seg-a"]),
        trace_times=np.array([0.0, 1.0], dtype=np.float32),
        title="Neuron split fields",
    )

    assert set(document.fields.keys()) == {
        NeuronDocumentBuilder.DISPLAY_FIELD_ID,
        NeuronDocumentBuilder.TRACE_FIELD_ID,
    }

    display_field = document.fields[NeuronDocumentBuilder.DISPLAY_FIELD_ID]
    trace_field = document.fields[NeuronDocumentBuilder.TRACE_FIELD_ID]
    morphology_view = document.views["morphology"]
    trace_view = document.views["trace"]

    assert display_field.dims == ("segment",)
    assert np.allclose(display_field.values, np.array([1.0, 2.0], dtype=np.float32))
    assert trace_field.dims == ("segment", "time")
    assert trace_field.coords["segment"].tolist() == ["seg-a"]
    assert morphology_view.color_field_id == NeuronDocumentBuilder.DISPLAY_FIELD_ID
    assert morphology_view.sample_dim is None
    assert trace_view.field_id == NeuronDocumentBuilder.TRACE_FIELD_ID


class DummyNeuronSession(NeuronSession):
    def build_sections(self):
        return []


def test_neuron_session_captures_new_trace_history_on_click():
    session = DummyNeuronSession()
    geometry = _geometry()
    voltage_values = np.array([1.0, 2.0], dtype=np.float32)

    session.geometry = geometry
    session._entity_index_by_id = {"seg-a": 0, "seg-b": 1}
    session._initialize_trace_history(0.0, voltage_values)

    session.handle(EntityClicked("seg-b"))
    updates = session.read_updates()

    assert len(updates) == 1
    assert isinstance(updates[0], FieldReplace)
    assert updates[0].field_id == NeuronDocumentBuilder.TRACE_FIELD_ID
    assert updates[0].coords["segment"].tolist() == ["seg-a", "seg-b"]
    assert updates[0].coords["time"].tolist() == [0.0]


def test_neuron_session_prefers_current_selected_entity_when_reinitializing_trace_history():
    session = DummyNeuronSession()
    geometry = _geometry()
    voltage_values = np.array([1.0, 2.0], dtype=np.float32)

    session.geometry = geometry
    session._entity_index_by_id = {"seg-a": 0, "seg-b": 1}
    session._ui_state["selected_entity_id"] = "seg-b"
    session._initialize_trace_history(0.0, voltage_values)

    segment_ids, times, values = session._trace_field_snapshot()

    assert segment_ids.tolist() == ["seg-b"]
    assert times.tolist() == [0.0]
    assert np.allclose(values, np.array([[2.0]], dtype=np.float32))


def test_neuron_session_prefers_selected_trace_entity_ids_when_reinitializing_trace_history():
    session = DummyNeuronSession()
    geometry = _geometry()
    voltage_values = np.array([1.0, 2.0], dtype=np.float32)

    session.geometry = geometry
    session._entity_index_by_id = {"seg-a": 0, "seg-b": 1}
    session._ui_state["selected_trace_entity_ids"] = ["seg-b", "seg-a"]
    session._initialize_trace_history(0.0, voltage_values)

    segment_ids, times, values = session._trace_field_snapshot()

    assert segment_ids.tolist() == ["seg-b", "seg-a"]
    assert times.tolist() == [0.0]
    assert np.allclose(values, np.array([[2.0], [1.0]], dtype=np.float32))
