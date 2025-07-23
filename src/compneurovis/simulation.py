from abc import ABC, abstractmethod
from typing import TypedDict
import numpy as np
from collections.abc import Callable

from multiprocessing.connection import Connection

class MorphologyMeta(TypedDict):
    positions:    np.ndarray   # (M,3) float32
    orientations: np.ndarray   # (M,3,3) float32
    radii:        np.ndarray   # (M,)   float32
    lengths:      np.ndarray   # (M,)   float32
    colors:       np.ndarray   # (M,4)  float32
    sec_names:    list[str]
    sec_idx:      np.ndarray   # (M,)   int32
    xloc:         np.ndarray   # (M,)   float32


class Simulation(ABC):

    def __init__(self):
        self.morphology_meta = None
        pass
    
    @abstractmethod
    def build_morphology_meta(self) -> MorphologyMeta:
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def record(self):
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def get_data(self, *args, **kwargs):
        pass
    
    @property
    def morphology_count(self):
        if self.morphology_meta is None:
            raise ValueError("Cannot get size of morphology before generating morphology file via build_morphology_meta")
        return len(self.morphology_meta["sec_idx"])

    def prepare(self):
        self.setup()
        self.morphology_meta = self.build_morphology_meta()
        self.record()


def simulation_process(sim: Simulation, data_pipe: Connection, cmd_pipe: Connection):
    try:
        sim.prepare()
        data_pipe.send(sim.morphology_meta)
        sim.initialize()

        while True:
            sim.step()
            while cmd_pipe.poll():
                if cmd_pipe.recv()=="reset":
                    sim.initialize()
            data = sim.get_data()
            ## TODO: More generic data sending
            data_pipe.send((data['t'], data['v']))
    finally:
        data_pipe.close()
        cmd_pipe.close()
                
        

