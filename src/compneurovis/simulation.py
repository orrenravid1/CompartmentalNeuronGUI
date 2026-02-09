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

    # --- Controllable parameter API ---
    def controllable_parameters(self) -> dict:
        """Return a dict mapping parameter names to a small spec dict describing
        how the viewer should render a control for it. Default is an empty dict.

        Spec keys (suggested):
        - type: 'float'|'int'|'bool'|'enum'
        - min, max: numeric bounds for sliders/spinboxes
        - steps: integer resolution for sliders
        - default: default value
        - label: human readable label
        - options: list (for enum)
        """
        return {}

    def apply_control(self, name: str, value) -> bool:
        """Apply a control change inside the simulation process.
        Default implementation sets an attribute if present. Returns True
        on success, False otherwise.
        """
        try:
            setattr(self, name, value)
            return True
        except Exception:
            return False

    def apply_action(self, name: str, payload) -> bool:
        """Apply an immediate action inside the simulation process.
        Default does nothing and returns False. Subclasses may implement
        actions such as creating an IClamp.
        """
        return False


def simulation_process(sim: Simulation, data_pipe: Connection, cmd_pipe: Connection):
    try:
        sim.prepare()
        data_pipe.send(sim.morphology_meta)
        sim.initialize()

        while True:
            sim.step()
            while cmd_pipe.poll():
                cmd = cmd_pipe.recv()
                # Backwards-compatible string command
                if cmd == "reset":
                    sim.initialize()
                # Tuple-based control commands: ("control", name, value)
                elif isinstance(cmd, tuple) and len(cmd) >= 3:
                    kind = cmd[0]
                    _name = cmd[1]
                    _val = cmd[2]
                    # New preferred API: ask simulation to apply the control
                    if kind == "control":
                        try:
                            ok = sim.apply_control(_name, _val)
                        except Exception:
                            ok = False
                        if not ok:
                            # fallback to setattr
                            try:
                                setattr(sim, _name, _val)
                            except Exception:
                                pass
                    # Preserve older "set" tuples
                    elif kind == "set":
                        try:
                            setattr(sim, _name, _val)
                        except Exception:
                            pass
                    # Action commands: ("action", name, payload)
                    elif kind == "action":
                        try:
                            sim.apply_action(_name, _val)
                        except Exception:
                            pass
                    # Special-case common NEURON parameter updates
                    try:
                        from neuron import h
                        if _name == 'dt':
                            h.dt = _val
                    except Exception:
                        pass
            data = sim.get_data()
            ## TODO: More generic data sending
            data_pipe.send((data['t'], data['v']))
    finally:
        data_pipe.close()
        cmd_pipe.close()
                
        

