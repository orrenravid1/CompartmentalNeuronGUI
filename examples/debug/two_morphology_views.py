from __future__ import annotations

import math
import time

import numpy as np

from compneurovis import AppSpec, Field, LayoutSpec, MorphologyGeometry, MorphologyViewSpec, PanelSpec, Scene, run_app
from compneurovis.session import BufferedSession, FieldReplace


DISPLAY_FIELD_ID = "morphology-display"


def build_geometry() -> MorphologyGeometry:
    return MorphologyGeometry(
        id="morphology-geometry",
        positions=np.array(
            [
                [-30.0, 0.0, 0.0],
                [-10.0, 0.0, 10.0],
                [10.0, 0.0, 22.0],
                [25.0, 0.0, 34.0],
            ],
            dtype=np.float32,
        ),
        orientations=np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 4, axis=0),
        radii=np.array([3.0, 2.5, 2.0, 1.5], dtype=np.float32),
        lengths=np.array([20.0, 18.0, 16.0, 14.0], dtype=np.float32),
        entity_ids=("seg-0", "seg-1", "seg-2", "seg-3"),
        section_names=("demo", "demo", "demo", "demo"),
        xlocs=np.array([0.12, 0.37, 0.63, 0.88], dtype=np.float32),
        labels=("demo@0.12", "demo@0.37", "demo@0.63", "demo@0.88"),
    )


def display_values(phase: float) -> np.ndarray:
    offsets = np.array([0.0, 0.8, 1.6, 2.4], dtype=np.float32)
    values = -65.0 + 55.0 * np.sin(phase + offsets) + 18.0 * np.cos((phase * 0.6) - offsets)
    return values.astype(np.float32)


def build_scene() -> Scene:
    geometry = build_geometry()
    field = Field(
        id=DISPLAY_FIELD_ID,
        values=display_values(0.0),
        dims=("segment",),
        coords={"segment": np.asarray(geometry.entity_ids)},
        unit="mV",
    )
    return Scene(
        fields={field.id: field},
        geometries={geometry.id: geometry},
        views={
            "morphology-left": MorphologyViewSpec(
                id="morphology-left",
                title="Morphology View A",
                geometry_id=geometry.id,
                color_field_id=field.id,
                background_color="#fbfbfb",
            ),
            "morphology-right": MorphologyViewSpec(
                id="morphology-right",
                title="Morphology View B",
                geometry_id=geometry.id,
                color_field_id=field.id,
                background_color="#f4f7fb",
            ),
        },
        layout=LayoutSpec(
            title="Two Morphology Views",
            panels=(
                PanelSpec(id="left-host", kind="view_3d", view_ids=("morphology-left",), title="Morphology View A"),
                PanelSpec(id="right-host", kind="view_3d", view_ids=("morphology-right",), title="Morphology View B"),
            ),
            panel_grid=(("left-host", "right-host"),),
        ),
    )


class AnimatedTwoMorphologyViewsSession(BufferedSession):
    def __init__(self, *, update_delay_s: float = 0.08):
        super().__init__()
        self.update_delay_s = update_delay_s
        self.phase = 0.0

    def initialize(self) -> Scene:
        return build_scene()

    def advance(self) -> None:
        time.sleep(self.update_delay_s)
        self.phase += 0.18
        self.emit(
            FieldReplace(
                field_id=DISPLAY_FIELD_ID,
                values=display_values(self.phase),
            )
        )

    def handle(self, command) -> None:
        del command


if __name__ == "__main__":
    run_app(
        AppSpec(
            session=AnimatedTwoMorphologyViewsSession,
            title="Two Morphology Views",
        )
    )
