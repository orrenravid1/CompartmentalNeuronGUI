import numpy as np
import pytest

from compneurovis.backends.neuron.scene import NeuronSceneBuilder
from compneurovis.backends.neuron.session import NeuronSession
from compneurovis.core import MorphologyGeometry
from compneurovis.backends.neuron.utils import (
    define_shape_layout,
    export_section_json,
    generate_layout,
    import_section_json,
    load_swc_multi,
    load_swc_neuron,
    parse_swc,
)
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


def test_neuron_scene_builder_splits_display_and_trace_fields():
    scene = NeuronSceneBuilder.build_scene(
        geometry=_geometry(),
        display_values=np.array([1.0, 2.0], dtype=np.float32),
        trace_values=np.array([[1.0, 2.0]], dtype=np.float32),
        trace_segment_ids=np.array(["seg-a"]),
        trace_times=np.array([0.0, 1.0], dtype=np.float32),
        title="Neuron split fields",
    )

    assert set(scene.fields.keys()) == {
        NeuronSceneBuilder.DISPLAY_FIELD_ID,
        NeuronSceneBuilder.HISTORY_FIELD_ID,
    }

    display_field = scene.fields[NeuronSceneBuilder.DISPLAY_FIELD_ID]
    trace_field = scene.fields[NeuronSceneBuilder.HISTORY_FIELD_ID]
    morphology_view = scene.views["morphology"]
    trace_view = scene.views["trace"]

    assert display_field.dims == ("segment",)
    assert np.allclose(display_field.values, np.array([1.0, 2.0], dtype=np.float32))
    assert trace_field.dims == ("segment", "time")
    assert trace_field.coords["segment"].tolist() == ["seg-a"]
    assert morphology_view.color_field_id == NeuronSceneBuilder.DISPLAY_FIELD_ID
    assert morphology_view.sample_dim is None
    assert morphology_view.color_map == "scalar"
    assert morphology_view.color_norm == "auto"
    assert trace_view.field_id == NeuronSceneBuilder.HISTORY_FIELD_ID


def test_neuron_backend_utils_package_exports_expected_helpers():
    assert define_shape_layout is not None
    assert export_section_json is not None
    assert generate_layout is not None
    assert import_section_json is not None
    assert load_swc_multi is not None
    assert load_swc_neuron is not None
    assert parse_swc is not None


class DummyNeuronSession(NeuronSession):
    def build_sections(self):
        return []


def test_neuron_session_record_registration_validates_names(monkeypatch):
    session = DummyNeuronSession()
    monkeypatch.setattr(session, "_rebuild_recorded_ptrs", lambda: None)

    ref = object()
    session.record("gate", ref)

    assert session._recorded_names == ["gate"]
    assert session._recorded_refs == [ref]

    with pytest.raises(ValueError, match="same length"):
        session.record_many(("a", "b"), (object(),))
    with pytest.raises(ValueError, match="unique"):
        session.record_many(("a", "a"), (object(), object()))
    with pytest.raises(ValueError, match="already records"):
        session.record("gate", object())


def test_neuron_session_captures_new_trace_history_on_click():
    session = DummyNeuronSession()
    geometry = _geometry()
    display_values = np.array([1.0, 2.0], dtype=np.float32)

    session.geometry = geometry
    session._entity_index_by_id = {"seg-a": 0, "seg-b": 1}
    session._initialize_trace_history(0.0, display_values)

    session.handle(EntityClicked("seg-b"))
    updates = session.read_updates()

    assert len(updates) == 1
    assert isinstance(updates[0], FieldReplace)
    assert updates[0].field_id == NeuronSceneBuilder.HISTORY_FIELD_ID
    assert updates[0].coords["segment"].tolist() == ["seg-a", "seg-b"]
    assert updates[0].coords["time"].tolist() == [0.0]


def test_neuron_session_prefers_current_selected_entity_when_reinitializing_trace_history():
    session = DummyNeuronSession()
    geometry = _geometry()
    display_values = np.array([1.0, 2.0], dtype=np.float32)

    session.geometry = geometry
    session._entity_index_by_id = {"seg-a": 0, "seg-b": 1}
    session._ui_state["selected_entity_id"] = "seg-b"
    session._initialize_trace_history(0.0, display_values)

    segment_ids, times, values = session._trace_field_snapshot()

    assert segment_ids.tolist() == ["seg-b"]
    assert times.tolist() == [0.0]
    assert np.allclose(values, np.array([[2.0]], dtype=np.float32))


def test_neuron_session_prefers_selected_trace_entity_ids_when_reinitializing_trace_history():
    session = DummyNeuronSession()
    geometry = _geometry()
    display_values = np.array([1.0, 2.0], dtype=np.float32)

    session.geometry = geometry
    session._entity_index_by_id = {"seg-a": 0, "seg-b": 1}
    session._ui_state["selected_trace_entity_ids"] = ["seg-b", "seg-a"]
    session._initialize_trace_history(0.0, display_values)

    segment_ids, times, values = session._trace_field_snapshot()

    assert segment_ids.tolist() == ["seg-b", "seg-a"]
    assert times.tolist() == [0.0]
    assert np.allclose(values, np.array([[2.0], [1.0]], dtype=np.float32))


def test_neuron_session_default_scene_uses_fixed_morphology_color_limits():
    session = DummyNeuronSession()
    geometry = _geometry()

    scene = session.build_scene(
        geometry=geometry,
        display_values=np.array([1.0, 2.0], dtype=np.float32),
        time_value=0.0,
    )

    assert scene.views["morphology"].color_map == "scalar"
    assert scene.views["morphology"].color_limits == (-80.0, 50.0)
    assert scene.views["morphology"].color_norm == "auto"


def test_neuron_session_default_scene_exposes_reset_action():
    session = DummyNeuronSession()
    geometry = _geometry()

    scene = session.build_scene(
        geometry=geometry,
        display_values=np.array([1.0, 2.0], dtype=np.float32),
        time_value=0.0,
    )

    assert "reset" in scene.actions
    assert scene.actions["reset"].label == "Reset"
    assert scene.actions["reset"].shortcuts == ("Space",)
    controls_panel = scene.layout.panel("controls-panel")
    assert controls_panel is not None
    assert controls_panel.action_ids == ("reset",)
