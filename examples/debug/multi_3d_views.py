from __future__ import annotations

import numpy as np

from compneurovis import AppSpec, Field, LayoutSpec, MorphologyGeometry, MorphologyViewSpec, Scene, SurfaceViewSpec, View3DHostSpec, grid_field, run_app


def build_demo_app() -> AppSpec:
    morphology_geometry = MorphologyGeometry(
        id="morphology-geometry",
        positions=np.array(
            [
                [-20.0, 0.0, 0.0],
                [0.0, 0.0, 10.0],
                [20.0, 0.0, 20.0],
            ],
            dtype=np.float32,
        ),
        orientations=np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 3, axis=0),
        radii=np.array([3.0, 2.0, 1.5], dtype=np.float32),
        lengths=np.array([18.0, 18.0, 18.0], dtype=np.float32),
        entity_ids=("seg-0", "seg-1", "seg-2"),
        section_names=("demo", "demo", "demo"),
        xlocs=np.array([0.17, 0.5, 0.83], dtype=np.float32),
        labels=("demo@0.17", "demo@0.50", "demo@0.83"),
    )
    morphology_field = Field(
        id="morphology-display",
        values=np.array([-65.0, -25.0, 10.0], dtype=np.float32),
        dims=("segment",),
        coords={"segment": np.array(morphology_geometry.entity_ids)},
        unit="mV",
    )

    x = np.linspace(-3.0, 3.0, 41, dtype=np.float32)
    y = np.linspace(-3.0, 3.0, 41, dtype=np.float32)
    z = (np.sin(x[None, :] * 1.5) + np.cos(y[:, None] * 1.5)).astype(np.float32)
    surface_field, surface_geometry = grid_field(field_id="surface-display", values=z, x_coords=x, y_coords=y)

    scene = Scene(
        fields={
            morphology_field.id: morphology_field,
            surface_field.id: surface_field,
        },
        geometries={
            morphology_geometry.id: morphology_geometry,
            surface_geometry.id: surface_geometry,
        },
        views={
            "morphology": MorphologyViewSpec(
                id="morphology",
                title="Morphology",
                geometry_id=morphology_geometry.id,
                color_field_id=morphology_field.id,
            ),
            "surface": SurfaceViewSpec(
                id="surface",
                title="Surface",
                field_id=surface_field.id,
                geometry_id=surface_geometry.id,
            ),
        },
        layout=LayoutSpec(
            title="Multi 3D View Demo",
            view_3d_hosts=(
                View3DHostSpec(id="morphology-host", view_ids=("morphology",)),
                View3DHostSpec(id="surface-host", view_ids=("surface",)),
            ),
        ),
    )
    return AppSpec(scene=scene, title="Multi 3D View Demo")


if __name__ == "__main__":
    run_app(build_demo_app())
