# CompartmentalNeuronGUI
A repository to create a new interactive 3d GUI for compartmental neuron modeling tools like NEURON, Jaxley, and MOOSE.
## Instructions
1. Install the relevant packages
2. Run visualizer.py
3. You can swap out the swc file being used by simply adding your swc file path instead of the one being used at the bottom of the file.
4. The demo will run a NEURON simulation where we load the morphology, add hodgkin huxley and passive dynamics, apply a number of current clamps over an 80 ms period
## Controls
1. You can interact by clicking on a section to watch its voltage trace
2. You can rotate around by clicking and dragging LMB
3. You can move around by pressing Shift and clicking and dragging LMB
4. You can zoom either with the scroll wheel or clicking and dragging RMB
## Goals:
Provide a generic way for someone to swap out their compartmental neuron simulator, simulation setup, and add their own widgets and interactions to their visualization.
