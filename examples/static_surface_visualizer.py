import numpy as np

from compneurovis.morphology_vis import run_visualizer
from compneurovis.static_visualization import StaticSurfaceSimulation


class InteractiveSurfaceSimulation(StaticSurfaceSimulation):
    COLOR_OPTIONS = [
        "black",
        "white",
        "gray",
        "red",
        "green",
        "blue",
        "orange",
        "purple",
    ]

    def __init__(self, x, y, z):
        self._surface_x = np.asarray(x, dtype=np.float32)
        self._surface_y = np.asarray(y, dtype=np.float32)
        self._surface_z = np.asarray(z, dtype=np.float32)

        self.axis_color = "black"
        self.text_color = "black"
        self.background_color = "white"
        self.surface_alpha = 0.9
        self.axis_alpha = 1.0
        self.tick_count = 7
        self.tick_length_scale = 1.0
        self.tick_label_size = 12.0
        self.axis_label_size = 16.0

        super().__init__(
            x=self._surface_x,
            y=self._surface_y,
            z=self._surface_z,
            title="interactive sinc surface",
            color_by="height",
            cmap="fire",
            render_axes=True,
            axes_in_middle=True,
            tick_count=self.tick_count,
            tick_length_scale=self.tick_length_scale,
            tick_label_size=self.tick_label_size,
            axis_label_size=self.axis_label_size,
            axis_color=self.axis_color,
            axis_labels=("x", "y", "height"),
            background_color=self.background_color,
            surface_alpha=self.surface_alpha,
            axis_alpha=self.axis_alpha,
        )
        self._sync_payload()

    def _sync_payload(self):
        payload = dict(self.initial_payload or {})
        payload.update(
            {
                "x": self._surface_x,
                "y": self._surface_y,
                "z": self._surface_z,
                "axis_color": self.axis_color,
                "text_color": self.text_color,
                "background_color": self.background_color,
                "surface_alpha": float(self.surface_alpha),
                "axis_alpha": float(self.axis_alpha),
                "tick_count": int(self.tick_count),
                "tick_length_scale": float(self.tick_length_scale),
                "tick_label_size": float(self.tick_label_size),
                "axis_label_size": float(self.axis_label_size),
            }
        )
        self.queue_scene_payload_update(payload)

    def build_initial_payload(self):
        self._sync_payload()
        return self.initial_payload

    def controllable_parameters(self) -> dict:
        return {
            "axis_color": {
                "type": "enum",
                "label": "Axis color",
                "options": self.COLOR_OPTIONS,
                "default": self.axis_color,
            },
            "text_color": {
                "type": "enum",
                "label": "Text color",
                "options": self.COLOR_OPTIONS,
                "default": self.text_color,
            },
            "background_color": {
                "type": "enum",
                "label": "Background",
                "options": self.COLOR_OPTIONS,
                "default": self.background_color,
            },
            "surface_alpha": {
                "type": "float",
                "label": "Surface alpha",
                "min": 0.1,
                "max": 1.0,
                "steps": 90,
                "default": self.surface_alpha,
            },
            "tick_count": {
                "type": "int",
                "label": "Axis ticks",
                "min": 0,
                "max": 12,
                "default": self.tick_count,
            },
            "tick_length_scale": {
                "type": "float",
                "label": "Tick length",
                "min": 0.0,
                "max": 3.0,
                "steps": 120,
                "default": self.tick_length_scale,
            },
            "tick_label_size": {
                "type": "float",
                "label": "Tick text size",
                "min": 6.0,
                "max": 24.0,
                "steps": 90,
                "default": self.tick_label_size,
            },
            "axis_label_size": {
                "type": "float",
                "label": "Axis label size",
                "min": 8.0,
                "max": 32.0,
                "steps": 96,
                "default": self.axis_label_size,
            },
            "axis_alpha": {
                "type": "float",
                "label": "Axes alpha",
                "min": 0.0,
                "max": 1.0,
                "steps": 100,
                "default": self.axis_alpha,
            },
        }

    def apply_control(self, name: str, value) -> bool:
        if name == "surface_alpha":
            self.surface_alpha = float(value)
        elif name == "axis_alpha":
            self.axis_alpha = float(value)
        elif name == "tick_count":
            self.tick_count = int(value)
        elif name == "tick_length_scale":
            self.tick_length_scale = float(value)
        elif name == "tick_label_size":
            self.tick_label_size = float(value)
        elif name == "axis_label_size":
            self.axis_label_size = float(value)
        elif name in {"axis_color", "text_color", "background_color"}:
            setattr(self, name, str(value))
        else:
            return super().apply_control(name, value)

        self._sync_payload()
        return True


x = np.linspace(-3.0, 3.0, 120)
y = np.linspace(-3.0, 3.0, 120)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Z = np.sinc(R) * 2.0

run_visualizer(InteractiveSurfaceSimulation(x=X, y=Y, z=Z))
