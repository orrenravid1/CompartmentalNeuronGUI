import numpy as np
import pytest

from compneurovis.core import GridGeometry, MorphologyGeometry


def test_grid_geometry_requires_matching_dim_keys():
    with pytest.raises(ValueError):
        GridGeometry(
            id="grid",
            dims=("x", "y"),
            coords={"x": np.arange(5, dtype=np.float32)},
        )


def test_morphology_geometry_requires_consistent_lengths():
    with pytest.raises(ValueError):
        MorphologyGeometry(
            id="morph",
            positions=np.zeros((2, 3), dtype=np.float32),
            orientations=np.zeros((2, 3, 3), dtype=np.float32),
            radii=np.ones(2, dtype=np.float32),
            lengths=np.ones(2, dtype=np.float32),
            entity_ids=("a",),
            section_names=("sec-a", "sec-b"),
            xlocs=np.array([0.1, 0.9], dtype=np.float32),
        )


def test_morphology_geometry_entity_info_returns_section_and_xloc():
    geometry = MorphologyGeometry(
        id="morph",
        positions=np.zeros((1, 3), dtype=np.float32),
        orientations=np.eye(3, dtype=np.float32)[None, :, :],
        radii=np.ones(1, dtype=np.float32),
        lengths=np.ones(1, dtype=np.float32),
        entity_ids=("sec-a@0.5",),
        section_names=("sec-a",),
        xlocs=np.array([0.5], dtype=np.float32),
        labels=("sec-a@0.5",),
    )

    info = geometry.entity_info("sec-a@0.5")

    assert info["section_name"] == "sec-a"
    assert info["xloc"] == 0.5
    assert info["label"] == "sec-a@0.5"
