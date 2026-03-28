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

