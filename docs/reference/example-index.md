---
title: Example Index
summary: Generated catalog of runnable examples grouped by backend and workflow.
---

# Example Index

This generated index groups runnable examples by backend and workflow and
extracts a short summary from each example when available.

## Live Simulation Backends

### NEURON

- **C. elegans morphology visualizer**: loads a directory of multi-tree SWC files as a single session. `python examples/neuron/c_elegans_visualizer.py` (`examples/neuron/c_elegans_visualizer.py`)
- **Multi-cell network visualizer**: three procedurally-built cells connected by synapses. `python examples/neuron/multicell_example.py` (`examples/neuron/multicell_example.py`)
- **Live NEURON visualizer**: minimal example of a single-cell live session loaded from an SWC file. `python examples/neuron/visualizer_example.py` (`examples/neuron/visualizer_example.py`)

### Jaxley

- **Multi-cell Jaxley visualizer**: three procedurally-built cells connected by synapses. `python examples/jaxley/multicell_example.py` (`examples/jaxley/multicell_example.py`)

## Custom Sessions and Solvers

### Custom

- **Custom FitzHugh-Nagumo backend**: complete example of a pure BufferedSession backend with its own fixed-step RK4 solver, explicit scene assembly, and bound controls/actions. No NEURON or Jaxley helper is involved. `python examples/custom/fitzhugh_nagumo_backend.py` (`examples/custom/fitzhugh_nagumo_backend.py`)

## Field and Surface Workflows

### Static / Interactive

- **Static surface visualizer**: renders a 3-D surface from a 2-D height field with interactive axes and appearance controls. No simulation or session is involved; the surface is computed once at startup and the UI controls update only the visual properties (colors, transparency, axis style). `python examples/surface_plot/static_surface_visualizer.py` (`examples/surface_plot/static_surface_visualizer.py`)
- **Surface cross-section visualizer**: renders a 3-D height field with a moveable cutting plane and a linked line plot showing the curve along that slice. Two controls let you choose the slice axis (x or y) and drag the cutting position; both the surface overlay and the line plot update together in real time. No simulation or session is involved. `python examples/surface_plot/surface_cross_section_visualizer.py` (`examples/surface_plot/surface_cross_section_visualizer.py`)

### Live

- **Animated surface**: live computation approach. Renders the same radially-expanding sinc wave as animated_surface_replay.py, but computes each frame on demand inside advance() rather than pre-computing them all at startup. A speed control changes the wave propagation rate in real time. `python examples/surface_plot/animated_surface_live.py` (`examples/surface_plot/animated_surface_live.py`)

### Replay

- **Animated surface**: replay approach. Renders a radially-expanding sinc wave by cycling through a pre-computed list of frames. Each step the session emits a FieldReplace with the next frame's values; the surface updates in place without rebuilding geometry or axes. `python examples/surface_plot/animated_surface_replay.py` (`examples/surface_plot/animated_surface_replay.py`)

## Debug and Architecture Probes

### Debug

- **Multi 3D Views**: Debug-oriented example for probing layout, session, or rendering behavior. `python examples/debug/multi_3d_views.py` (`examples/debug/multi_3d_views.py`)
- **Session Error After Open**: Debug-oriented example for probing layout, session, or rendering behavior. `python examples/debug/session_error_after_open.py` (`examples/debug/session_error_after_open.py`)
- **Two Line Plots**: debug-oriented example that renders two live line plots at once with no 3-D host. `python examples/debug/two_line_plots.py` (`examples/debug/two_line_plots.py`)
- **Two Morphology Views**: Debug-oriented example for probing layout, session, or rendering behavior. `python examples/debug/two_morphology_views.py` (`examples/debug/two_morphology_views.py`)

