import numpy as np

from compneurovis.simulation import Simulation


class StaticVisualizationSimulation(Simulation):
    """Base class for simulations that only provide static viewer content."""

    def __init__(self, initial_payload=None, data=None):
        super().__init__()
        self.initial_payload = initial_payload
        self._pending_scene_payload = None
        self.data = {'t': 0.0}
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
        background_color=None,
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
        if background_color is not None:
            payload['background_color'] = background_color
        super().__init__(initial_payload=payload, data=data)
