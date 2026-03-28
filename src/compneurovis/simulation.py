from abc import ABC, abstractmethod
import time

from multiprocessing.connection import Connection

class Simulation(ABC):
    def __init__(self):
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

    def build_initial_payload(self):
        """Return optional static viewer payload generated after setup().

        Subclasses can override this to provide any precomputed metadata a
        viewer may want, such as morphology geometry. The base simulation does
        not assume any specific payload shape.
        """
        return None

    def consume_scene_payload_update(self):
        """Return a pending scene payload update for the viewer, if any."""
        return None

    def is_live(self) -> bool:
        """Whether the simulation advances continuously in the worker process."""
        return True

    def idle_sleep(self) -> float:
        """Sleep interval used when the worker is not advancing live state."""
        return 0.05

    def prepare(self):
        self.setup()
        initial_payload = self.build_initial_payload()
        self.record()
        return initial_payload

    def close(self):
        pass

    def on_reset(self):
        """Called in the simulation process when a reset is requested.
        Default reinitializes simulation state. Override to add extra teardown
        or re-wiring before/after initialize."""
        self.initialize()

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

    # --- Viewer interaction hooks (called in GUI process) ---
    def on_control_gui(self, name: str, value, viewer=None) -> None:
        """Called in the GUI process when a control value changes.
        Override to track GUI-side state (e.g. current iclamp amplitude)."""
        pass

    def handle_key_press(self, key, viewer) -> bool:
        """Handle a key press in the viewer. Return True if consumed."""
        return False

    def handle_key_release(self, key, viewer) -> bool:
        """Handle a key release in the viewer. Return True if consumed."""
        return False

    def handle_segment_click(self, sec: str, xloc: float, viewer) -> bool:
        """Handle a segment click in the viewer.
        Return True to suppress the default trace-select behaviour."""
        return False

    def handle_viewer_reset(self, viewer) -> None:
        """Called in the GUI process after a simulation reset and trace clear.
        Override to customize plot axis state after reset."""
        vb = viewer.plot2d.getPlotItem().getViewBox()
        vb.enableAutoRange(x=True, y=True)
        vb.setLimits(yMin=None, yMax=None)


def simulation_process(sim: Simulation, data_pipe: Connection, cmd_pipe: Connection):
    try:
        initial_payload = sim.prepare()
        data_pipe.send(("initial_payload", initial_payload))
        sim.initialize()

        while True:
            if sim.is_live():
                sim.step()
            else:
                time.sleep(sim.idle_sleep())
            while cmd_pipe.poll():
                cmd = cmd_pipe.recv()
                # Backwards-compatible string command
                if cmd == "reset":
                    sim.on_reset()
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

            scene_payload = sim.consume_scene_payload_update()
            if scene_payload is not None:
                if (
                    isinstance(scene_payload, tuple)
                    and len(scene_payload) == 2
                    and isinstance(scene_payload[0], str)
                ):
                    data_pipe.send(scene_payload)
                else:
                    data_pipe.send(("scene_payload", scene_payload))

            data = sim.get_data()
            if data is not None:
                data_pipe.send(("data", data))
    finally:
        data_pipe.close()
        cmd_pipe.close()
                
        
