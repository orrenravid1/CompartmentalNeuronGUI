from __future__ import annotations

import numpy as np

from compneurovis.backends.jaxley import JaxleySceneBuilder, JaxleySession
from compneurovis.core import MorphologyGeometry
from compneurovis.session import EntityClicked, FieldReplace


def test_jaxley_scene_builder_splits_display_and_trace_fields():
    geometry = MorphologyGeometry(
        id="morphology",
        positions=np.zeros((2, 3), dtype=np.float32),
        orientations=np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0),
        radii=np.ones(2, dtype=np.float32),
        lengths=np.ones(2, dtype=np.float32),
        entity_ids=("seg-a", "seg-b"),
        section_names=("sec-a", "sec-b"),
        xlocs=np.array([0.25, 0.75], dtype=np.float32),
        labels=("sec-a@0.25", "sec-b@0.75"),
    )

    scene = JaxleySceneBuilder.build_scene(
        geometry=geometry,
        display_values=np.array([1.0, 2.0], dtype=np.float32),
        trace_values=np.array([[1.0], [2.0]], dtype=np.float32),
        trace_segment_ids=np.array(["seg-a", "seg-b"]),
        trace_times=np.array([0.0], dtype=np.float32),
    )

    display_field = scene.fields[JaxleySceneBuilder.DISPLAY_FIELD_ID]
    trace_field = scene.fields[JaxleySceneBuilder.HISTORY_FIELD_ID]

    assert display_field.dims == ("segment",)
    assert np.allclose(display_field.values, np.array([1.0, 2.0], dtype=np.float32))
    assert trace_field.dims == ("segment", "time")
    assert trace_field.coords["segment"].tolist() == ["seg-a", "seg-b"]
    assert trace_field.coords["time"].tolist() == [0.0]
    morphology_view = scene.views["morphology"]
    assert morphology_view.color_field_id == JaxleySceneBuilder.DISPLAY_FIELD_ID
    assert morphology_view.sample_dim is None
    assert morphology_view.color_map == "scalar"
    assert morphology_view.color_norm == "auto"
    trace_view = scene.views["trace"]
    assert trace_view.field_id == JaxleySceneBuilder.HISTORY_FIELD_ID


class DummyJaxleySession(JaxleySession):
    def build_cells(self):
        raise AssertionError("build_cells should not be called in this unit test")


def test_jaxley_session_build_scene_uses_sparse_trace_history_contract():
    session = DummyJaxleySession()
    geometry = MorphologyGeometry(
        id="morphology",
        positions=np.zeros((2, 3), dtype=np.float32),
        orientations=np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0),
        radii=np.ones(2, dtype=np.float32),
        lengths=np.ones(2, dtype=np.float32),
        entity_ids=("seg-a", "seg-b"),
        section_names=("sec-a", "sec-b"),
        xlocs=np.array([0.25, 0.75], dtype=np.float32),
        labels=("sec-a@0.25", "sec-b@0.75"),
    )
    display_values = np.array([1.0, 2.0], dtype=np.float32)
    session.geometry = geometry
    session._entity_index_by_id = {"seg-a": 0, "seg-b": 1}
    session._initialize_trace_history(0.0, display_values)

    scene = session.build_scene(
        geometry=geometry,
        display_values=display_values,
        time_value=0.0,
    )

    trace_field = scene.fields[JaxleySceneBuilder.HISTORY_FIELD_ID]
    assert trace_field.coords["segment"].tolist() == ["seg-a"]
    assert trace_field.coords["time"].tolist() == [0.0]


def test_jaxley_session_captures_new_trace_history_on_click():
    session = DummyJaxleySession()
    geometry = MorphologyGeometry(
        id="morphology",
        positions=np.zeros((2, 3), dtype=np.float32),
        orientations=np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0),
        radii=np.ones(2, dtype=np.float32),
        lengths=np.ones(2, dtype=np.float32),
        entity_ids=("seg-a", "seg-b"),
        section_names=("sec-a", "sec-b"),
        xlocs=np.array([0.25, 0.75], dtype=np.float32),
        labels=("sec-a@0.25", "sec-b@0.75"),
    )
    display_values = np.array([1.0, 2.0], dtype=np.float32)
    session.geometry = geometry
    session._entity_index_by_id = {"seg-a": 0, "seg-b": 1}
    session._initialize_trace_history(0.0, display_values)

    session.handle(EntityClicked("seg-b"))
    updates = session.read_updates()

    assert len(updates) == 1
    assert isinstance(updates[0], FieldReplace)
    assert updates[0].field_id == JaxleySceneBuilder.HISTORY_FIELD_ID
    assert updates[0].coords["segment"].tolist() == ["seg-a", "seg-b"]
    assert updates[0].coords["time"].tolist() == [0.0]


def test_jaxley_session_prefers_current_selected_entity_when_reinitializing_trace_history():
    session = DummyJaxleySession()
    geometry = MorphologyGeometry(
        id="morphology",
        positions=np.zeros((2, 3), dtype=np.float32),
        orientations=np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0),
        radii=np.ones(2, dtype=np.float32),
        lengths=np.ones(2, dtype=np.float32),
        entity_ids=("seg-a", "seg-b"),
        section_names=("sec-a", "sec-b"),
        xlocs=np.array([0.25, 0.75], dtype=np.float32),
        labels=("sec-a@0.25", "sec-b@0.75"),
    )
    display_values = np.array([1.0, 2.0], dtype=np.float32)
    session.geometry = geometry
    session._entity_index_by_id = {"seg-a": 0, "seg-b": 1}
    session._ui_state["selected_entity_id"] = "seg-b"
    session._initialize_trace_history(0.0, display_values)

    segment_ids, times, values = session._trace_field_snapshot()

    assert segment_ids.tolist() == ["seg-b"]
    assert times.tolist() == [0.0]
    assert np.allclose(values, np.array([[2.0]], dtype=np.float32))


def test_jaxley_session_prefers_selected_trace_entity_ids_when_reinitializing_trace_history():
    session = DummyJaxleySession()
    geometry = MorphologyGeometry(
        id="morphology",
        positions=np.zeros((2, 3), dtype=np.float32),
        orientations=np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0),
        radii=np.ones(2, dtype=np.float32),
        lengths=np.ones(2, dtype=np.float32),
        entity_ids=("seg-a", "seg-b"),
        section_names=("sec-a", "sec-b"),
        xlocs=np.array([0.25, 0.75], dtype=np.float32),
        labels=("sec-a@0.25", "sec-b@0.75"),
    )
    display_values = np.array([1.0, 2.0], dtype=np.float32)
    session.geometry = geometry
    session._entity_index_by_id = {"seg-a": 0, "seg-b": 1}
    session._ui_state["selected_trace_entity_ids"] = ["seg-b", "seg-a"]
    session._initialize_trace_history(0.0, display_values)

    segment_ids, times, values = session._trace_field_snapshot()

    assert segment_ids.tolist() == ["seg-b", "seg-a"]
    assert times.tolist() == [0.0]
    assert np.allclose(values, np.array([[2.0], [1.0]], dtype=np.float32))


def test_jaxley_session_default_scene_uses_generic_auto_morphology_coloring():
    session = DummyJaxleySession()
    geometry = MorphologyGeometry(
        id="morphology",
        positions=np.zeros((2, 3), dtype=np.float32),
        orientations=np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0),
        radii=np.ones(2, dtype=np.float32),
        lengths=np.ones(2, dtype=np.float32),
        entity_ids=("seg-a", "seg-b"),
        section_names=("sec-a", "sec-b"),
        xlocs=np.array([0.25, 0.75], dtype=np.float32),
        labels=("sec-a@0.25", "sec-b@0.75"),
    )
    session.geometry = geometry
    session._entity_index_by_id = {"seg-a": 0, "seg-b": 1}
    session._initialize_trace_history(0.0, np.array([1.0, 2.0], dtype=np.float32))

    scene = session.build_scene(
        geometry=geometry,
        display_values=np.array([1.0, 2.0], dtype=np.float32),
        time_value=0.0,
    )

    assert scene.views["morphology"].color_map == "scalar"
    assert scene.views["morphology"].color_norm == "auto"


def test_jaxley_session_default_scene_exposes_reset_action():
    session = DummyJaxleySession()
    geometry = MorphologyGeometry(
        id="morphology",
        positions=np.zeros((2, 3), dtype=np.float32),
        orientations=np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0),
        radii=np.ones(2, dtype=np.float32),
        lengths=np.ones(2, dtype=np.float32),
        entity_ids=("seg-a", "seg-b"),
        section_names=("sec-a", "sec-b"),
        xlocs=np.array([0.25, 0.75], dtype=np.float32),
        labels=("sec-a@0.25", "sec-b@0.75"),
    )
    session.geometry = geometry
    session._entity_index_by_id = {"seg-a": 0, "seg-b": 1}
    session._initialize_trace_history(0.0, np.array([1.0, 2.0], dtype=np.float32))

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
