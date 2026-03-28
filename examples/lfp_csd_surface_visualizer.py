from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import numpy as np

from compneurovis import SurfaceViewSpec, build_surface_app, grid_field, run_app


DEFAULT_PKL = (
    Path(__file__).resolve().parents[2]
    / "NeuronVisualization"
    / "26march17_NMDAR25reduction_LFP.pkl"
)

DEFAULT_ISI_MS = [150, 300, 600, 1200, 2400, 3000, 6000]
DEFAULT_SAMPLE_RATE_HZ = 10000
DEFAULT_WINDOW_MS = 150.0
DEFAULT_STIM_START_MS = 3000.0
DEFAULT_NUM_STIMS = 15
DEFAULT_AVG_LAST_N = 5
DEFAULT_DEPTH_UM = 2000.0


def load_run(path: Path) -> dict:
    with path.open("rb") as fh:
        return pickle.load(fh)


def extract_lfps_by_experiment(run: dict) -> list[np.ndarray]:
    if "simData" in run:
        return [np.asarray(d["LFP"], dtype=np.float32) for d in run["simData"]]
    if "LFP" in run:
        return [np.asarray(arr, dtype=np.float32) for arr in run["LFP"]]
    raise KeyError(f"Expected 'simData' or 'LFP' in pickle, found keys: {list(run.keys())}")


def to_channels_by_time(lfp: np.ndarray) -> np.ndarray:
    arr = np.asarray(lfp, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D LFP array, got shape {arr.shape}")
    if arr.shape[0] > arr.shape[1]:
        return arr.T
    return arr


def compute_csd(lfp: np.ndarray, vaknin: bool = True) -> np.ndarray:
    chans_by_time = to_channels_by_time(lfp)
    padded = np.vstack([chans_by_time[0:1], chans_by_time, chans_by_time[-1:]]) if vaknin else chans_by_time
    return -(padded[:-2] - 2.0 * padded[1:-1] + padded[2:])


def stimulus_times_ms(
    isi_ms: int,
    start_ms: float = DEFAULT_STIM_START_MS,
    num_stims: int = DEFAULT_NUM_STIMS,
) -> list[float]:
    return [start_ms + isi_ms * idx for idx in range(num_stims)]


def average_erp_csd_for_experiment(
    lfp: np.ndarray,
    isi_ms: int,
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
    window_ms: float = DEFAULT_WINDOW_MS,
    avg_last_n: int = DEFAULT_AVG_LAST_N,
) -> np.ndarray:
    csd = compute_csd(lfp, vaknin=True)
    window_samples = int(sample_rate_hz * window_ms / 1000.0)
    starts_ms = stimulus_times_ms(isi_ms)[-avg_last_n:]

    avg = np.zeros((csd.shape[0], window_samples), dtype=np.float32)
    for t_ms in starts_ms:
        start_idx = int(sample_rate_hz * t_ms / 1000.0)
        end_idx = start_idx + window_samples
        avg += csd[:, start_idx:end_idx]
    avg /= float(len(starts_ms))
    return avg


def build_surface_app_from_pickle(pkl_path: Path, experiment_index: int):
    run = load_run(pkl_path)
    lfps = extract_lfps_by_experiment(run)
    if not 0 <= experiment_index < len(lfps):
        raise IndexError(f"Experiment index {experiment_index} out of range for {len(lfps)} runs")

    lfp = lfps[experiment_index]
    avg_csd = average_erp_csd_for_experiment(lfp=lfp, isi_ms=DEFAULT_ISI_MS[experiment_index])

    time_ms = np.linspace(0.0, DEFAULT_WINDOW_MS, avg_csd.shape[1], dtype=np.float32)
    depths_um = np.linspace(0.0, DEFAULT_DEPTH_UM, avg_csd.shape[0], dtype=np.float32)

    field, geometry = grid_field(
        field_id="csd",
        values=avg_csd,
        x_coords=time_ms,
        y_coords=depths_um,
        x_dim="time",
        y_dim="channel",
    )
    title = f"Average ERP CSD surface (ISI {DEFAULT_ISI_MS[experiment_index]} ms, run {experiment_index})"
    surface_view = SurfaceViewSpec(
        id="surface",
        title=title,
        field_id=field.id,
        geometry_id=geometry.id,
        axis_labels=("time", "channel", "csd"),
    )
    return build_surface_app(field=field, geometry=geometry, title=title, surface_view=surface_view)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one experiment from an LFP pickle as a static CSD surface."
    )
    parser.add_argument("--pkl", type=Path, default=DEFAULT_PKL)
    parser.add_argument("--experiment-index", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_app(build_surface_app_from_pickle(args.pkl.expanduser().resolve(), args.experiment_index))
