from neuron import h, gui
from src.neuronutils.swc_utils import load_swc_multi, load_swc_neuron
import os

swc_path = os.path.join("..","res","celegans_cells_swc")
swc_files = [f for f in os.listdir(swc_path)]
allsecs = []
for i,swcf in enumerate(swc_files):
    cell_name = swcf.split('.')[0]
    trees = load_swc_multi(os.path.join(swc_path, swcf), cell_name)
    seclists = trees.values()
    for seclist in seclists:
        for sec in seclist:
            allsecs.append(sec)
print(allsecs)