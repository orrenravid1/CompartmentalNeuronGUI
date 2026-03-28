import numpy as np
import pytest

from compneurovis.core import Field


def test_field_validates_dim_coord_consistency():
    with pytest.raises(ValueError):
        Field(
            id="bad",
            values=np.zeros((2, 3), dtype=np.float32),
            dims=("x", "y"),
            coords={"x": np.arange(2, dtype=np.float32)},
        )


def test_field_named_dim_slicing_keeps_metadata():
    field = Field(
        id="voltage",
        values=np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32),
        dims=("segment", "time"),
        coords={
            "segment": np.array(["seg-a", "seg-b"]),
            "time": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        },
        unit="mV",
    )

    sliced = field.select({"segment": "seg-b"})

    assert sliced.dims == ("time",)
    assert np.allclose(sliced.coord("time"), np.array([0.0, 1.0, 2.0], dtype=np.float32))
    assert np.allclose(sliced.values, np.array([10.0, 20.0, 30.0], dtype=np.float32))
    assert sliced.unit == "mV"


def test_field_numeric_selector_uses_nearest_coordinate():
    field = Field(
        id="csd",
        values=np.arange(12, dtype=np.float32).reshape(3, 4),
        dims=("channel", "time"),
        coords={
            "channel": np.array([100.0, 200.0, 300.0], dtype=np.float32),
            "time": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        },
    )

    sliced = field.select({"channel": 210.0})
    assert sliced.dims == ("time",)
    assert np.allclose(sliced.values, np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32))


def test_field_label_list_selector_keeps_dimension_and_order():
    field = Field(
        id="voltage",
        values=np.arange(8, dtype=np.float32).reshape(2, 4),
        dims=("segment", "time"),
        coords={
            "segment": np.array(["seg-a", "seg-b"]),
            "time": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        },
    )

    sliced = field.select({"segment": ["seg-b", "seg-a"]})

    assert sliced.dims == ("segment", "time")
    assert sliced.coord("segment").tolist() == ["seg-b", "seg-a"]
    assert np.allclose(sliced.values[0], np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32))


def test_field_append_extends_named_dimension_and_trims():
    field = Field(
        id="voltage",
        values=np.array([[1.0, 2.0], [10.0, 20.0]], dtype=np.float32),
        dims=("segment", "time"),
        coords={
            "segment": np.array(["seg-a", "seg-b"]),
            "time": np.array([0.0, 1.0], dtype=np.float32),
        },
        unit="mV",
    )

    appended = field.append(
        "time",
        np.array([[3.0, 4.0], [30.0, 40.0]], dtype=np.float32),
        np.array([2.0, 3.0], dtype=np.float32),
        max_length=3,
    )

    assert appended.dims == ("segment", "time")
    assert appended.coord("time").tolist() == [1.0, 2.0, 3.0]
    assert np.allclose(
        appended.values,
        np.array([[2.0, 3.0, 4.0], [20.0, 30.0, 40.0]], dtype=np.float32),
    )
    assert appended.unit == "mV"
