"""
Nonvisual parameter sweep for the Jaxley multicell example.

This isolates propagation in the feedforward chain without launching the UI.
The sweep uses a single early pulse so the run time stays practical while still
testing whether cell1 can drive cell2 and cell3 through the synapses.

Run:
    python experiments/jaxley_multicell_param_sweep.py
"""

from __future__ import annotations

from itertools import product
import json
from pathlib import Path
import sys
import time

import numpy as np
import jax

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compneurovis.session import Reset


def load_multicell_session_class():
    example_path = ROOT / "examples" / "jaxley" / "multicell_example.py"
    text = example_path.read_text(encoding="utf-8")
    text = text.replace(
        '\nif __name__ == "__main__":\n    run_app(build_jaxley_app(MultiCellSession))\n',
        "\n",
    )
    namespace: dict[str, object] = {}
    exec(compile(text, str(example_path), "exec"), namespace)
    return namespace["MultiCellSession"]


MultiCellSession = load_multicell_session_class()


class SweepSession(MultiCellSession):
    def _pulse_schedule(self):
        return [(2.0, self.stim_dur, self.stim_amp)]


def soma_entity_ids(session: SweepSession) -> list[str]:
    chosen: list[str] = []
    for cell_name in ("cell1", "cell2", "cell3"):
        matches = [
            entity_id
            for entity_id in session.geometry.entity_ids
            if session.geometry.entity_info(entity_id)["section_name"] == f"{cell_name}_branch_0"
        ]
        if not matches:
            raise RuntimeError(f"Could not find branch_0 entities for {cell_name}")
        chosen.append(min(matches, key=lambda entity_id: abs(session.geometry.entity_info(entity_id)["xloc"] - 0.5)))
    return chosen


def run_point(session: SweepSession, soma_indices: list[int], *, syn_gs: float, syn_v_th: float, syn_delta: float, steps: int) -> dict[str, object]:
    session.apply_control("syn_gs", syn_gs)
    session.apply_control("syn_v_th", syn_v_th)
    session.apply_control("syn_delta", syn_delta)
    session.handle(Reset())
    session.read_updates()

    max_v = np.full(len(soma_indices), -np.inf, dtype=np.float32)
    for _ in range(steps):
        session.advance()
        max_v = np.maximum(max_v, session._last_voltage_values[soma_indices])
    session.read_updates()

    score = float(min(max_v[1], max_v[2]))
    return {
        "syn_gs": syn_gs,
        "syn_v_th": syn_v_th,
        "syn_delta": syn_delta,
        "cell1_max_v": float(max_v[0]),
        "cell2_max_v": float(max_v[1]),
        "cell3_max_v": float(max_v[2]),
        "cell2_spiked": bool(max_v[1] > 0.0),
        "cell3_spiked": bool(max_v[2] > 0.0),
        "score": score,
    }


def main() -> int:
    syn_gs_values = [5e-4, 1e-3, 3e-3, 1e-2]
    syn_v_th_values = [-45.0, -35.0, -25.0]
    syn_delta_values = [3.0, 10.0]
    steps = 300

    session = SweepSession()
    t0 = time.perf_counter()
    session.initialize()
    init_s = time.perf_counter() - t0

    soma_ids = soma_entity_ids(session)
    soma_indices = [session._entity_index_by_id[entity_id] for entity_id in soma_ids]

    results = []
    for syn_gs, syn_v_th, syn_delta in product(syn_gs_values, syn_v_th_values, syn_delta_values):
        point = run_point(
            session,
            soma_indices,
            syn_gs=syn_gs,
            syn_v_th=syn_v_th,
            syn_delta=syn_delta,
            steps=steps,
        )
        results.append(point)
        print(
            json.dumps(
                {
                    "syn_gs": syn_gs,
                    "syn_v_th": syn_v_th,
                    "syn_delta": syn_delta,
                    "cell2_max_v": point["cell2_max_v"],
                    "cell3_max_v": point["cell3_max_v"],
                    "score": point["score"],
                }
            )
        )

    ranked = sorted(results, key=lambda row: row["score"], reverse=True)
    summary = {
        "init_seconds": init_s,
        "steps": steps,
        "soma_ids": soma_ids,
        "num_points": len(results),
        "cell2_spike_count": sum(1 for row in results if row["cell2_spiked"]),
        "cell3_spike_count": sum(1 for row in results if row["cell3_spiked"]),
        "top_results": ranked[:10],
    }
    print("SUMMARY")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
