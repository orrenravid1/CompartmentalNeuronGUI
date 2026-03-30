from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from compneurovis.jaxleyutils import parse_swc


ROOT = Path(__file__).resolve().parents[1]
CELEGANS_DIR = ROOT / "res" / "celegans_cells_swc"
pytestmark = pytest.mark.jaxley


def _run_python(code: str) -> dict:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = completed.stdout.strip().splitlines()
    return json.loads(stdout[-1])


def test_parse_swc_finds_single_root_for_small_fixture():
    nodes, children = parse_swc(str(CELEGANS_DIR / "VA12.swc"))

    roots = [node_id for node_id, node in nodes.items() if node["parent"] < 0]

    assert len(nodes) > 0
    assert len(children) > 0
    assert roots == [1]


def test_load_swc_jaxley_small_celegans_files():
    for filename in ["VA12.swc", "RIGR.swc", "AIAR.swc"]:
        result = _run_python(
            f"""
            import json
            import jax
            from compneurovis.jaxleyutils import load_swc_jaxley

            cell = load_swc_jaxley(r"{CELEGANS_DIR / filename}", ncomp=4, min_radius=1.0)
            print(json.dumps({{
                "meta_name": getattr(cell, "meta_name", None),
                "node_count": len(cell.nodes),
                "branch_count": len(cell.xyzr),
                "has_cols": all(col in cell.nodes.columns for col in ["x", "y", "z", "radius", "length"]),
            }}))
            """
        )

        assert result["meta_name"] == Path(filename).stem
        assert result["node_count"] > 0
        assert result["branch_count"] > 0
        assert result["has_cols"] is True


def test_jaxley_swc_cache_roundtrip_matches_raw_load(tmp_path: Path):
    for filename in ["VA12.swc", "RIGR.swc", "AIAR.swc"]:
        cache_path = tmp_path / f"{Path(filename).stem}.pkl"
        result = _run_python(
            f"""
            import json
            import numpy as np
            import jax
            from compneurovis.jaxleyutils import (
                load_cached_swc_jaxley,
                load_swc_jaxley,
                save_swc_jaxley_cache,
            )

            swc_path = r"{CELEGANS_DIR / filename}"
            cache_path = r"{cache_path}"
            save_swc_jaxley_cache(swc_path, cache_path, ncomp=4, min_radius=1.0)

            raw_cell = load_swc_jaxley(swc_path, ncomp=4, min_radius=1.0)
            cached_cell = load_cached_swc_jaxley(cache_path)

            raw_values = raw_cell.nodes[["x", "y", "z", "radius", "length"]].to_numpy(np.float32)
            cached_values = cached_cell.nodes[["x", "y", "z", "radius", "length"]].to_numpy(np.float32)

            print(json.dumps({{
                "meta_name": getattr(cached_cell, "meta_name", None),
                "same_node_count": len(raw_cell.nodes) == len(cached_cell.nodes),
                "same_branch_count": len(raw_cell.xyzr) == len(cached_cell.xyzr),
                "same_values": bool(np.allclose(raw_values, cached_values, rtol=1e-5, atol=1e-5)),
            }}))
            """
        )

        assert result["meta_name"] == Path(filename).stem
        assert result["same_node_count"] is True
        assert result["same_branch_count"] is True
        assert result["same_values"] is True


def test_jaxley_programmatic_geometry_has_no_compartment_gaps():
    example_path = ROOT / "examples" / "jaxley" / "multicell_example.py"
    result = _run_python(
        f"""
        import json
        import numpy as np
        import jax
        from compneurovis.backends.jaxley.document import JaxleyDocumentBuilder

        example_path = r"{example_path}"
        text = open(example_path, "r", encoding="utf-8").read()
        text = text.rsplit("run_app(build_jaxley_app(MultiCellSession))", 1)[0]
        ns = {{}}
        exec(compile(text, example_path, "exec"), ns)
        MultiCellSession = ns["MultiCellSession"]

        session = MultiCellSession()
        cells = session.build_cells()
        network = session.build_network(cells)
        session.setup_model(network, cells)
        geometry = JaxleyDocumentBuilder.build_morphology_geometry(
            network.nodes,
            xyzr=network.xyzr,
            cell_names=session.cell_names(cells),
        )

        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        max_gap = 0.0
        for branch_name in sorted(set(geometry.section_names)):
            idxs = [i for i, name in enumerate(geometry.section_names) if name == branch_name]
            for i in range(len(idxs) - 1):
                left = idxs[i]
                right = idxs[i + 1]
                left_axis = geometry.orientations[left] @ z_axis
                right_axis = geometry.orientations[right] @ z_axis
                left_end = geometry.positions[left] + 0.5 * geometry.lengths[left] * left_axis
                right_start = geometry.positions[right] - 0.5 * geometry.lengths[right] * right_axis
                max_gap = max(max_gap, float(np.linalg.norm(right_start - left_end)))

        print(json.dumps({{"max_gap": max_gap}}))
        """
    )

    assert result["max_gap"] < 1e-3
