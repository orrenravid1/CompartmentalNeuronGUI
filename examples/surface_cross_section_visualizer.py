import numpy as np

from compneurovis.morphology_vis import run_visualizer
from compneurovis.static_visualization import StaticSurfaceSimulation


class SurfaceCrossSectionSimulation(StaticSurfaceSimulation):
    def __init__(self, x, y, z):
        self._surface_x = np.asarray(x, dtype=np.float32)
        self._surface_y = np.asarray(y, dtype=np.float32)
        self._surface_z = np.asarray(z, dtype=np.float32)

        self.slice_axis = "x"
        self.slice_position = 0.0

        super().__init__(
            x=self._surface_x,
            y=self._surface_y,
            z=self._surface_z,
            title="surface cross-section viewer",
            color_by="height",
            cmap="fire",
            render_axes=True,
            axes_in_middle=True,
            tick_count=7,
            axis_color="black",
            text_color="black",
            axis_labels=("x", "y", "height"),
            background_color="white",
            surface_alpha=0.9,
            axis_alpha=0.95,
            tick_length_scale=1.0,
            tick_label_size=12.0,
            axis_label_size=16.0,
        )
        self._sync_payload()

    def _axis_values(self, axis: str) -> np.ndarray:
        if axis == "y":
            return self._surface_y[:, 0]
        return self._surface_x[0, :]

    def _slice_index_and_value(self):
        axis_values = self._axis_values(self.slice_axis)
        idx = int(round(self.slice_position * (len(axis_values) - 1)))
        idx = max(0, min(len(axis_values) - 1, idx))
        return idx, float(axis_values[idx])

    def _cross_section_payload(self, idx: int, axis_value: float) -> dict:
        if self.slice_axis == "x":
            xvals = self._surface_y[:, idx]
            yvals = self._surface_z[:, idx]
            xlabel = "y"
            title = f"Cross section at x = {axis_value:.3f}"
        else:
            xvals = self._surface_x[idx, :]
            yvals = self._surface_z[idx, :]
            xlabel = "x"
            title = f"Cross section at y = {axis_value:.3f}"

        return {
            "x": np.asarray(xvals, dtype=np.float32),
            "y": np.asarray(yvals, dtype=np.float32),
            "xlabel": xlabel,
            "ylabel": "height",
            "title": title,
            "pen": "#1f3c88",
            "background_color": "white",
        }

    def _sync_payload(self):
        idx, axis_value = self._slice_index_and_value()
        payload = dict(self.initial_payload or {})
        payload.update(
            {
                "x": self._surface_x,
                "y": self._surface_y,
                "z": self._surface_z,
                "slice_rect": {
                    "axis": self.slice_axis,
                    "value": axis_value,
                    "color": "#111111",
                    "alpha": 0.95,
                    "width": 3.0,
                },
                "plot2d": self._cross_section_payload(idx, axis_value),
            }
        )
        self.queue_scene_payload_update(payload)

    def build_initial_payload(self):
        self._sync_payload()
        return self.initial_payload

    def controllable_parameters(self) -> dict:
        return {
            "slice_axis": {
                "type": "enum",
                "label": "Slice axis",
                "options": ["x", "y"],
                "default": self.slice_axis,
            },
            "slice_position": {
                "type": "float",
                "label": "Slice position",
                "min": 0.0,
                "max": 1.0,
                "steps": 200,
                "default": self.slice_position,
            },
        }

    def on_control_gui(self, name: str, value, viewer=None) -> None:
        if name != "slice_axis" or viewer is None:
            return
        self.slice_axis = str(value)
        self.slice_position = 0.0
        slider = viewer._controls.get("slice_position")
        if slider is not None:
            slider.setValue(0)

    def apply_control(self, name: str, value) -> bool:
        if name == "slice_axis":
            self.slice_axis = str(value)
            self.slice_position = 0.0
        elif name == "slice_position":
            self.slice_position = float(value)
        else:
            return super().apply_control(name, value)

        self._sync_payload()
        return True


def build_demo_surface():
    x = np.linspace(-4.0, 4.0, 180, dtype=np.float32)
    y = np.linspace(-3.0, 3.0, 160, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    Z = (
        0.9 * np.sin(1.4 * X)
        + 0.35 * X
        + 1.1 * np.cos(0.9 * Y)
        + 0.45 * np.sin(1.8 * Y + 0.5 * X)
        + 0.08 * X * Y
    ).astype(np.float32)
    return X, Y, Z


if __name__ == "__main__":
    X, Y, Z = build_demo_surface()
    run_visualizer(SurfaceCrossSectionSimulation(x=X, y=Y, z=Z))
