import numpy as np

from compneurovis.morphology_vis import run_visualizer
from compneurovis.static_visualization import StaticSurfaceSimulation


x = np.linspace(-3.0, 3.0, 120)
y = np.linspace(-3.0, 3.0, 120)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Z = np.sinc(R) * 2.0

run_visualizer(
    StaticSurfaceSimulation(
        x=X,
        y=Y,
        z=Z,
        title="sinc surface",
    )
)
