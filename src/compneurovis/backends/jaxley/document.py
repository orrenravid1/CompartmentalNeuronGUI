from __future__ import annotations

import numpy as np

from compneurovis.core import Document, Field, LayoutSpec, LinePlotViewSpec, MorphologyGeometry, MorphologyViewSpec, StateBinding


class JaxleyDocumentBuilder:
    DISPLAY_FIELD_ID = "voltage_display"
    TRACE_FIELD_ID = "voltage_trace"

    @staticmethod
    def _split_xyzr_into_equal_length_segments(xyzr: np.ndarray, ncomp: int) -> list[np.ndarray]:
        if len(xyzr) == 1:
            return [xyzr] * ncomp

        xyz = xyzr[:, :3]
        deltas = np.diff(xyz, axis=0)
        dists = np.linalg.norm(deltas, axis=1)
        cum_dists = np.concatenate([[0.0], np.cumsum(dists)])
        total_length = cum_dists[-1]
        target_dists = np.linspace(0.0, total_length, ncomp + 1)

        idxs = np.searchsorted(cum_dists, target_dists, side="right") - 1
        idxs = np.clip(idxs, 0, len(xyz) - 2)
        local_dist = target_dists - cum_dists[idxs]
        dists = np.where(dists < 1e-14, 1e-14, dists)
        segment_lens = dists[idxs]
        frac = (local_dist / segment_lens)[:, None]
        split_points = xyzr[idxs] + frac * (xyzr[idxs + 1] - xyzr[idxs])

        segments = []
        all_points = [split_points[0]]
        for i in range(1, len(split_points)):
            mask = (cum_dists > target_dists[i - 1]) & (cum_dists < target_dists[i])
            between_points = xyzr[mask]
            segment = np.vstack([all_points[-1], *between_points, split_points[i]])
            segments.append(segment.astype(np.float32))
            all_points.append(split_points[i])
        return segments

    @staticmethod
    def _segment_radius(segment_xyzr: np.ndarray) -> float:
        if len(segment_xyzr) <= 1:
            return float(segment_xyzr[0, 3])
        lengths = np.linalg.norm(np.diff(segment_xyzr[:, :3], axis=0), axis=1)
        weights = np.zeros((len(segment_xyzr),), dtype=np.float32)
        weights[1:] += lengths
        weights[:-1] += lengths
        total = float(weights.sum())
        if total <= 1e-12:
            return float(np.mean(segment_xyzr[:, 3]))
        weights /= total
        return float(np.sum(segment_xyzr[:, 3] * weights))

    @staticmethod
    def build_morphology_geometry(
        nodes,
        *,
        xyzr: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
        cell_names: list[str] | tuple[str, ...] | None = None,
    ) -> MorphologyGeometry:
        ordered = nodes.sort_values("global_comp_index").reset_index(drop=True)
        if ordered.empty:
            raise ValueError("JaxleyDocumentBuilder requires at least one compartment")

        positions = ordered[["x", "y", "z"]].to_numpy(np.float32)
        lengths = np.maximum(ordered["length"].to_numpy(np.float32), 1e-6)
        radii = np.maximum(ordered["radius"].to_numpy(np.float32), 1e-6)
        directions = np.repeat(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), len(ordered), axis=0)

        for branch_idx, branch in ordered.groupby("global_branch_index", sort=False):
            idxs = branch.index.to_numpy()
            if xyzr is not None and int(branch_idx) < len(xyzr):
                branch_xyzr = np.asarray(xyzr[int(branch_idx)], dtype=np.float32)
                segments = JaxleyDocumentBuilder._split_xyzr_into_equal_length_segments(branch_xyzr, len(idxs))
                branch_positions = []
                branch_lengths = []
                branch_radii = []
                branch_dirs = []
                for segment in segments:
                    start = segment[0, :3]
                    end = segment[-1, :3]
                    diff = end - start
                    seg_length = float(np.linalg.norm(diff))
                    if seg_length <= 1e-6:
                        seg_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                        seg_length = 1e-6
                    else:
                        seg_dir = diff / seg_length
                    branch_positions.append(0.5 * (start + end))
                    branch_lengths.append(seg_length)
                    branch_radii.append(max(JaxleyDocumentBuilder._segment_radius(segment), 1e-6))
                    branch_dirs.append(seg_dir)
                positions[idxs] = np.asarray(branch_positions, dtype=np.float32)
                lengths[idxs] = np.asarray(branch_lengths, dtype=np.float32)
                radii[idxs] = np.asarray(branch_radii, dtype=np.float32)
                directions[idxs] = np.asarray(branch_dirs, dtype=np.float32)
            else:
                pts = branch[["x", "y", "z"]].to_numpy(np.float32)
                if len(idxs) == 1:
                    continue
                branch_dirs = np.zeros_like(pts)
                branch_dirs[:-1] = pts[1:] - pts[:-1]
                branch_dirs[-1] = pts[-1] - pts[-2]
                norms = np.linalg.norm(branch_dirs, axis=1, keepdims=True)
                nonzero = norms[:, 0] > 1e-6
                branch_dirs[nonzero] = branch_dirs[nonzero] / norms[nonzero]
                branch_dirs[~nonzero] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                directions[idxs] = branch_dirs

        cos_t = directions[:, 2]
        ang = np.arccos(np.clip(cos_t, -1.0, 1.0))
        ax = np.cross(np.repeat([[0.0, 0.0, 1.0]], len(directions), axis=0), directions)
        ax_n = np.linalg.norm(ax, axis=1, keepdims=True)
        ax_u = np.zeros_like(ax)
        np.divide(ax, ax_n, out=ax_u, where=ax_n > 1e-6)
        ux, uy, uz = ax_u.T

        k = np.zeros((len(directions), 3, 3), dtype=np.float32)
        k[:, 0, 1] = -uz
        k[:, 0, 2] = uy
        k[:, 1, 0] = uz
        k[:, 1, 2] = -ux
        k[:, 2, 0] = -uy
        k[:, 2, 1] = ux
        k2 = k @ k
        identity = np.eye(3, dtype=np.float32)[None, :, :]
        orientations = identity + np.sin(ang)[:, None, None] * k + (1.0 - cos_t)[:, None, None] * k2

        global_cell = ordered["global_cell_index"].to_numpy(np.int32)
        local_branch = ordered["local_branch_index"].to_numpy(np.int32)
        local_comp = ordered["local_comp_index"].to_numpy(np.float32)
        counts = ordered.groupby("global_branch_index")["global_comp_index"].transform("size").to_numpy(np.float32)
        xlocs = (local_comp + 0.5) / np.maximum(counts, 1.0)

        if cell_names is None:
            names = {int(cell_idx): f"cell_{int(cell_idx)}" for cell_idx in np.unique(global_cell)}
        else:
            names = {int(cell_idx): str(cell_names[int(cell_idx)]) for cell_idx in np.unique(global_cell)}
        section_names = tuple(f"{names[int(cell_idx)]}_branch_{int(branch_idx)}" for cell_idx, branch_idx in zip(global_cell, local_branch))
        labels = tuple(f"{section}@{float(xloc):.3f}" for section, xloc in zip(section_names, xlocs))

        return MorphologyGeometry(
            id="morphology",
            positions=positions.astype(np.float32),
            orientations=orientations.astype(np.float32),
            radii=radii.astype(np.float32),
            lengths=lengths.astype(np.float32),
            entity_ids=labels,
            section_names=section_names,
            xlocs=xlocs.astype(np.float32),
            labels=labels,
        )

    @staticmethod
    def build_document(
        *,
        geometry: MorphologyGeometry,
        display_values: np.ndarray,
        trace_values: np.ndarray,
        trace_segment_ids: np.ndarray,
        trace_times: np.ndarray,
        controls=None,
        actions=None,
        title: str = "CompNeuroVis",
        control_ids: tuple[str, ...] | None = None,
        action_ids: tuple[str, ...] | None = None,
    ) -> Document:
        display_field = Field(
            id=JaxleyDocumentBuilder.DISPLAY_FIELD_ID,
            values=np.asarray(display_values, dtype=np.float32),
            dims=("segment",),
            coords={
                "segment": np.asarray(geometry.entity_ids),
            },
            unit="mV",
        )
        trace_field = Field(
            id=JaxleyDocumentBuilder.TRACE_FIELD_ID,
            values=np.asarray(trace_values, dtype=np.float32),
            dims=("segment", "time"),
            coords={
                "segment": np.asarray(trace_segment_ids),
                "time": np.asarray(trace_times, dtype=np.float32),
            },
            unit="mV",
        )
        views = {
            "morphology": MorphologyViewSpec(
                id="morphology",
                title="Morphology",
                geometry_id=geometry.id,
                color_field_id=display_field.id,
                entity_dim="segment",
                sample_dim=None,
                color_map="voltage",
            ),
            "trace": LinePlotViewSpec(
                id="trace",
                title="Voltage",
                field_id=trace_field.id,
                x_dim="time",
                selectors={"segment": StateBinding("selected_entity_id")},
                x_label="Time",
                y_label="Voltage",
                x_unit="ms",
                y_unit="mV",
                pen="#1f3c88",
            ),
        }
        return Document(
            fields={display_field.id: display_field, trace_field.id: trace_field},
            geometries={geometry.id: geometry},
            views=views,
            controls={} if controls is None else dict(controls),
            actions={} if actions is None else dict(actions),
            layout=LayoutSpec(
                title=title,
                main_3d_view_id="morphology",
                line_plot_view_id="trace",
                control_ids=tuple(controls.keys()) if controls and control_ids is None else (() if control_ids is None else control_ids),
                action_ids=tuple(actions.keys()) if actions and action_ids is None else (() if action_ids is None else action_ids),
            ),
        )
