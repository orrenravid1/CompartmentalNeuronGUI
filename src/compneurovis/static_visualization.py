import numpy as np

from compneurovis.simulation import Simulation


class StaticVisualizationSimulation(Simulation):
    """Base class for simulations that only provide static viewer content."""

    def __init__(self, initial_payload=None, data=None):
        super().__init__()
        self.initial_payload = initial_payload
        self._pending_scene_payload = None
        self.data = {}
        if data:
            self.data.update(data)

    def setup(self):
        pass

    def record(self):
        pass

    def initialize(self):
        pass

    def step(self):
        pass

    def get_data(self, *args, **kwargs):
        if not args:
            return dict(self.data)
        return {k: v for k, v in self.data.items() if k in args}

    def build_initial_payload(self):
        return self.initial_payload

    def queue_scene_payload_update(self, payload):
        self.initial_payload = payload
        self._pending_scene_payload = payload

    def consume_scene_payload_update(self):
        payload = self._pending_scene_payload
        self._pending_scene_payload = None
        return payload

    def is_live(self) -> bool:
        return False


class StaticSurfaceSimulation(StaticVisualizationSimulation):
    """Visualization-only simulation for static 3D surface plots."""

    def __init__(
        self,
        x,
        y,
        z,
        colors=None,
        title="surface",
        data=None,
        color_by=None,
        cmap="bwr",
        clim=None,
        surface_alpha=1.0,
        background_color=None,
        render_axes=False,
        axes_in_middle=True,
        tick_count=None,
        tick_length_scale=1.0,
        tick_label_size=12.0,
        axis_label_size=16.0,
        axis_color=None,
        text_color=None,
        axis_alpha=1.0,
        axis_labels=None,
    ):
        payload = {
            'kind': 'surface',
            'x': np.asarray(x, dtype=np.float32),
            'y': np.asarray(y, dtype=np.float32),
            'z': np.asarray(z, dtype=np.float32),
            'title': title,
        }
        if colors is not None:
            payload['colors'] = np.asarray(colors, dtype=np.float32)
        if color_by is not None:
            payload['color_by'] = str(color_by)
        if cmap is not None:
            payload['cmap'] = str(cmap)
        if clim is not None:
            payload['clim'] = tuple(float(v) for v in clim)
        if surface_alpha is not None:
            payload['surface_alpha'] = float(surface_alpha)
        if background_color is not None:
            payload['background_color'] = background_color
        if render_axes is not None:
            payload['render_axes'] = bool(render_axes)
        if axes_in_middle is not None:
            payload['axes_in_middle'] = bool(axes_in_middle)
        if tick_count is not None:
            payload['tick_count'] = int(tick_count)
        if tick_length_scale is not None:
            payload['tick_length_scale'] = float(tick_length_scale)
        if tick_label_size is not None:
            payload['tick_label_size'] = float(tick_label_size)
        if axis_label_size is not None:
            payload['axis_label_size'] = float(axis_label_size)
        if axis_color is not None:
            payload['axis_color'] = axis_color
        if text_color is not None:
            payload['text_color'] = text_color
        if axis_alpha is not None:
            payload['axis_alpha'] = float(axis_alpha)
        if axis_labels is not None:
            payload['axis_labels'] = tuple(str(v) for v in axis_labels)
        super().__init__(initial_payload=payload, data=data)
