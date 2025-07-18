# CompartmentalNeuronGUI
A repository to create a new interactive 3d GUI for compartmental neuron modeling tools like NEURON, Jaxley, Arbor, and MOOSE.
<img width="902" alt="image" src="https://github.com/user-attachments/assets/fbdfcc50-5547-4f87-ae4d-48f3e3c653cd" />
<img width="900" height="474" alt="image" src="https://github.com/user-attachments/assets/3639a36b-59bd-48e3-a0f0-7d74a17f5a3b" />


## Current State:
Currently there is a functional demo both for the entire c elegans nervous system and a complex morphology of a single cell.
- `python visualizer.py` runs the complex morphology of a single cell demo
- `python c_elegans_vis_test.py` runs the whole c elegans nervous system demo

## Instructions
1. Install neuron via `pip install neuron` on Unix or via an exe for Windows here: [https://www.neuron.yale.edu/neuron/download](https://www.neuron.yale.edu/neuron/download)
2. Install the relevant packages from requirements.txt via `pip install -r requirements.txt`
3. Run `python visualizer.py`
4. You can swap out the swc file being used by simply adding your swc file path instead of the one being used at the bottom of the file.
5. The demo will run a NEURON simulation where we load the morphology, add Hodgkin Huxley and passive dynamics, apply a number of current clamps over an 80 ms period
6.
    a. You can update the experimental setup currently in the `neuron_process` method, though it has not been modularized yet to make things straightforward.

    b. The easy things to change currently are changing the mechanisms being added and point processes: IClamps, etc.
## Controls
1. You can interact by clicking on a section to watch its voltage trace
2. You can rotate around by clicking and dragging LMB
3. You can move around by pressing Shift and clicking and dragging LMB
4. You can zoom either with the scroll wheel or clicking and dragging RMB
5. Press Spacebar to restart the simulation.
## Goal:
Provide a generic way for someone to swap out their compartmental neuron simulator, simulation setup, and add their own widgets and interactions to their visualization.

# Known issues:
- PyOpenGL has a [known bug](https://github.com/mcfletch/pyopengl/issues/149) affecting Instanced Rendering when using `numpy>=2.3` so currently need to use `numpy<2.3` for things to work. But a [fix](https://github.com/mcfletch/pyopengl/pull/150) is in progress.
