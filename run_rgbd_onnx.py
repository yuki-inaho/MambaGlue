"""Run exported MambaGlue ONNX on saved RGB-D partial scenes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import cv2
import numpy as np
import onnxruntime as ort
import torch

from mambaglue import MambaGlue, SuperPoint
from mambaglue.onnx_export import pad_features_to_fixed
from mambaglue.utils import load_image


GRID_ROWS = 6
GRID_COLS = 8
PERPENDICULAR_ANGLE_DEG = 60.0
STATIC_MOTION_PX = 2.0
STRONG_TRANSVERSE_MOTION_PX = 2.0
MIN_PAIR_INLIERS = 8
MIN_REGION_INLIERS = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def point(value: np.ndarray | list[float]) -> list[float]:
    return [round(float(item), 3) for item in value]


def grid_region(x: float, y: float, width: int, height: int) -> str:
    col = min(GRID_COLS - 1, max(0, int(x / width * GRID_COLS)))
    row = min(GRID_ROWS - 1, max(0, int(y / height * GRID_ROWS)))
    return f"g{row:02d}_{col:02d}"


def region_bbox(region_id: str, width: int, height: int) -> list[float]:
    row = int(region_id[1:3])
    col = int(region_id[4:6])
    return [
        round(col * width / GRID_COLS, 3),
        round(row * height / GRID_ROWS, 3),
        round((col + 1) * width / GRID_COLS, 3),
        round((row + 1) * height / GRID_ROWS, 3),
    ]


def axis_metrics(vector: np.ndarray, axis: np.ndarray) -> dict[str, float | None]:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return {
            "angle_to_dominant_axis_deg": None,
            "parallel_component_px": 0.0,
            "perpendicular_component_px": 0.0,
        }
    cosine = float(np.clip(abs(vector @ axis) / norm, -1.0, 1.0))
    return {
        "angle_to_dominant_axis_deg": round(math.degrees(math.acos(cosine)), 3),
        "parallel_component_px": round(abs(float(vector @ axis)), 3),
        "perpendicular_component_px": round(
            abs(float(vector[0] * axis[1] - vector[1] * axis[0])), 3
        ),
    }


def add_observation(
    track: dict,
    frame: dict,
    keypoint_index: int,
    xy: np.ndarray | list[float],
    region_id: str,
) -> None:
    if (
        track["observations"]
        and track["observations"][-1]["frame_row"] == frame["frame_row"]
    ):
        return
    track["observations"].append(
        {
            "frame_row": frame["frame_row"],
            "frame_id": frame["frame_id"],
            "image_basename": frame["image_basename"],
            "keypoint_index": int(keypoint_index),
            "xy_px": point(xy),
            "region_id": region_id,
        }
    )


def finalize_tracks(tracks: dict[int, dict], axis: np.ndarray) -> list[dict]:
    result = []
    for track_id, track in sorted(tracks.items()):
        observations = track["observations"]
        points = np.asarray([item["xy_px"] for item in observations], dtype=np.float64)
        steps = np.diff(points, axis=0)
        norms = np.linalg.norm(steps, axis=1) if len(steps) else np.empty(0)
        angles = []
        for step in steps:
            metric = axis_metrics(step, axis)["angle_to_dominant_axis_deg"]
            if metric is not None:
                angles.append(metric)
        result.append(
            {
                "mambaglue_track_id": track_id,
                "observation_count": len(observations),
                "first_frame_row": observations[0]["frame_row"],
                "last_frame_row": observations[-1]["frame_row"],
                "median_step_px": None
                if not len(norms)
                else round(float(np.median(norms)), 3),
                "median_angle_to_dominant_axis_deg": None
                if not angles
                else round(float(np.median(angles)), 3),
                "stationary_step_fraction": None
                if not len(norms)
                else round(float(np.mean(norms <= STATIC_MOTION_PX)), 4),
                "perpendicular_step_fraction": None
                if not angles
                else round(
                    float(np.mean(np.asarray(angles) >= PERPENDICULAR_ANGLE_DEG)), 4
                ),
                "region_ids": sorted({item["region_id"] for item in observations}),
                "observations": observations,
            }
        )
    return result


class OnnxMambaGlue:
    def __init__(self, model_path: Path, fixed_keypoints: int) -> None:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(model_path.resolve()),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.fixed_keypoints = fixed_keypoints
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_names = [item.name for item in self.session.get_outputs()]

    def run(self, features0: dict, features1: dict) -> dict:
        fixed0 = pad_features_to_fixed(features0, self.fixed_keypoints)
        fixed1 = pad_features_to_fixed(features1, self.fixed_keypoints)
        inputs = {
            "keypoints0": fixed0["keypoints"].cpu().numpy().astype(np.float32),
            "descriptors0": fixed0["descriptors"].cpu().numpy().astype(np.float32),
            "image_size0": fixed0["image_size"].cpu().numpy().astype(np.float32),
            "valid0": fixed0["valid"].cpu().numpy(),
            "keypoints1": fixed1["keypoints"].cpu().numpy().astype(np.float32),
            "descriptors1": fixed1["descriptors"].cpu().numpy().astype(np.float32),
            "image_size1": fixed1["image_size"].cpu().numpy().astype(np.float32),
            "valid1": fixed1["valid"].cpu().numpy(),
        }
        started = time.perf_counter()
        values = self.session.run(self.output_names, inputs)
        runtime_ms = (time.perf_counter() - started) * 1000.0
        outputs = dict(zip(self.output_names, values, strict=True))
        outputs["fixed0"] = fixed0
        outputs["fixed1"] = fixed1
        outputs["runtime_ms"] = runtime_ms
        return outputs


def extract_features(
    extractor: SuperPoint,
    path: Path,
    *,
    resize: int,
) -> dict:
    image = load_image(path).cpu()
    with torch.inference_mode():
        return extractor.extract(image, resize=resize)


def geometric_pair(
    source_frame: dict,
    target_frame: dict,
    outputs: dict,
    axis: np.ndarray,
) -> tuple[dict, list[dict]]:
    fixed0 = outputs["fixed0"]
    fixed1 = outputs["fixed1"]
    matches0 = outputs["matches0"][0]
    scores0 = outputs["matching_scores0"][0]
    valid0 = fixed0["valid"][0].cpu().numpy().astype(bool)
    valid1 = fixed1["valid"][0].cpu().numpy().astype(bool)
    pairs = [
        (index0, int(index1), float(scores0[index0]))
        for index0, index1 in enumerate(matches0)
        if valid0[index0]
        and int(index1) >= 0
        and int(index1) < len(valid1)
        and valid1[int(index1)]
    ]
    keypoints0 = fixed0["keypoints"][0].cpu().numpy()
    keypoints1 = fixed1["keypoints"][0].cpu().numpy()
    source_points = np.asarray([keypoints0[i] for i, _, _ in pairs], dtype=np.float32)
    target_points = np.asarray([keypoints1[j] for _, j, _ in pairs], dtype=np.float32)
    base = {
        "from_row": source_frame["frame_row"],
        "to_row": target_frame["frame_row"],
        "from": source_frame["image_basename"],
        "to": target_frame["image_basename"],
        "matches": len(pairs),
        "match_score_median": None
        if not pairs
        else round(float(np.median([score for _, _, score in pairs])), 6),
        "onnx_runtime_ms": round(float(outputs["runtime_ms"]), 3),
        "status": "insufficient_matches",
    }
    if len(pairs) < 8:
        return base, []
    matrix, mask = cv2.estimateAffinePartial2D(
        source_points,
        target_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000,
        confidence=0.995,
        refineIters=10,
    )
    if matrix is None or mask is None:
        return base | {"status": "no_affine"}, []
    inlier = mask.reshape(-1).astype(bool)
    inlier_count = int(inlier.sum())
    if inlier_count < 6:
        return base | {"status": "weak_affine", "inliers": inlier_count}, []
    flow = target_points - source_points
    selected_flow = flow[inlier]
    vector = np.median(selected_flow, axis=0)
    point_norms = np.linalg.norm(selected_flow, axis=1)
    metrics = axis_metrics(vector, axis)
    height, width = source_frame["image_size_hw"]
    links = []
    for (index0, index1, score), source_xy, target_xy, keep in zip(
        pairs, source_points, target_points, inlier, strict=True
    ):
        if not keep:
            continue
        links.append(
            {
                "source_keypoint_index": index0,
                "target_keypoint_index": index1,
                "source_xy_px": point(source_xy),
                "target_xy_px": point(target_xy),
                "flow_px": point(target_xy - source_xy),
                "point_motion_px": round(float(np.linalg.norm(target_xy - source_xy)), 3),
                "matching_score": round(score, 6),
                "source_region_id": grid_region(
                    float(source_xy[0]), float(source_xy[1]), width, height
                ),
                "target_region_id": grid_region(
                    float(target_xy[0]), float(target_xy[1]), width, height
                ),
            }
        )
    grouped: dict[str, list[dict]] = defaultdict(list)
    for link in links:
        grouped[link["source_region_id"]].append(link)
    regions = []
    for region_id, region_links in sorted(grouped.items()):
        vectors = np.asarray([item["flow_px"] for item in region_links])
        region_vector = np.median(vectors, axis=0)
        region_norms = np.asarray([item["point_motion_px"] for item in region_links])
        region_metrics = axis_metrics(region_vector, axis)
        angle = region_metrics["angle_to_dominant_axis_deg"]
        regions.append(
            {
                "region_id": region_id,
                "bbox_xyxy_px": region_bbox(region_id, width, height),
                "inliers": len(region_links),
                "median_flow_px": point(region_vector),
                "median_point_motion_px": round(float(np.median(region_norms)), 3),
                **region_metrics,
                "stationary": bool(
                    len(region_links) >= MIN_REGION_INLIERS
                    and np.median(region_norms) <= STATIC_MOTION_PX
                ),
                "strong_transverse": bool(
                    len(region_links) >= MIN_REGION_INLIERS
                    and angle is not None
                    and angle >= PERPENDICULAR_ANGLE_DEG
                    and np.median(region_norms) >= STRONG_TRANSVERSE_MOTION_PX
                ),
                "mambaglue_track_ids": [],
            }
        )
    angle = metrics["angle_to_dominant_axis_deg"]
    base.update(
        {
            "status": "ok",
            "inliers": inlier_count,
            "inlier_ratio": round(inlier_count / len(pairs), 4),
            "affine_2x3": [[round(float(value), 7) for value in row] for row in matrix],
            "median_flow_px": point(vector),
            "median_flow_norm_px": round(float(np.linalg.norm(vector)), 3),
            "median_point_motion_px": round(float(np.median(point_norms)), 3),
            "p90_point_motion_px": round(float(np.percentile(point_norms, 90)), 3),
            **metrics,
            "direction_perpendicular": bool(
                angle is not None and angle >= PERPENDICULAR_ANGLE_DEG
            ),
            "stationary": bool(
                inlier_count >= MIN_PAIR_INLIERS
                and np.median(point_norms) <= STATIC_MOTION_PX
            ),
            "strong_transverse": bool(
                inlier_count >= MIN_PAIR_INLIERS
                and angle is not None
                and angle >= PERPENDICULAR_ANGLE_DEG
                and np.median(point_norms) >= STRONG_TRANSVERSE_MOTION_PX
            ),
            "regions": regions,
        }
    )
    return base, links


def pytorch_comparison(
    matcher: MambaGlue,
    features0: dict,
    features1: dict,
    outputs: dict,
) -> dict:
    with torch.inference_mode():
        expected = matcher({"image0": features0, "image1": features1})
    count0 = min(features0["keypoints"].shape[1], outputs["matches0"].shape[1])
    expected_matches = expected["matches0"][0, :count0].cpu().numpy()
    actual_matches = outputs["matches0"][0, :count0]
    expected_scores = expected["matching_scores0"][0, :count0].cpu().numpy()
    actual_scores = outputs["matching_scores0"][0, :count0]
    return {
        "compared_keypoints": int(count0),
        "match_index_mismatch_count": int(np.count_nonzero(expected_matches != actual_matches)),
        "matching_score_max_abs_error": round(
            float(np.max(np.abs(expected_scores - actual_scores))), 8
        ),
    }


def run(args: argparse.Namespace) -> dict:
    source = json.loads(args.source_analysis.read_text(encoding="utf-8"))
    os.environ.setdefault("TORCH_HOME", str(args.torch_home.resolve()))
    torch.set_num_threads(2)
    extractor = SuperPoint(max_num_keypoints=args.fixed_keypoints).eval().cpu()
    onnx_matcher = OnnxMambaGlue(args.model, args.fixed_keypoints)
    feature_cache: dict[int, dict] = {}
    pytorch_matcher = None
    next_track_id = 1
    scene_results = []

    for source_scene in source["scenes"]:
        axis = np.asarray(
            source_scene["normal_motion_axis"]["unit_vector_xy"], dtype=np.float64
        )
        axis /= np.linalg.norm(axis)
        frames = source_scene["frames"]
        tracks: dict[int, dict] = {}
        assignments: dict[int, dict[int, int]] = {frames[0]["frame_row"]: {}}
        pair_results = []
        source_pairs = {item["pair_id"]: item for item in source_scene["pair_results"]}
        candidate_pair_ids = set(source_scene["candidate_pair_ids"])

        for pair_index, (frame0, frame1) in enumerate(
            zip(frames[:-1], frames[1:], strict=True)
        ):
            for frame in (frame0, frame1):
                row = frame["frame_row"]
                if row not in feature_cache:
                    feature_cache[row] = extract_features(
                        extractor, args.data_root / frame["rgb_path"], resize=args.resize
                    )
            features0 = feature_cache[frame0["frame_row"]]
            features1 = feature_cache[frame1["frame_row"]]
            outputs = onnx_matcher.run(features0, features1)
            pair, links = geometric_pair(frame0, frame1, outputs, axis)
            pair_id = f"pair_{pair_index:03d}_{frame0['frame_row']}_{frame1['frame_row']}"
            pair["pair_id"] = pair_id

            source_assignment = assignments.setdefault(frame0["frame_row"], {})
            target_assignment: dict[int, int] = {}
            linked = []
            for link in links:
                source_index = link["source_keypoint_index"]
                target_index = link["target_keypoint_index"]
                track_id = source_assignment.get(source_index)
                if track_id is None:
                    track_id = next_track_id
                    next_track_id += 1
                    source_assignment[source_index] = track_id
                    tracks[track_id] = {"observations": []}
                    add_observation(
                        tracks[track_id],
                        frame0,
                        source_index,
                        link["source_xy_px"],
                        link["source_region_id"],
                    )
                target_assignment[target_index] = track_id
                add_observation(
                    tracks[track_id],
                    frame1,
                    target_index,
                    link["target_xy_px"],
                    link["target_region_id"],
                )
                linked.append({"mambaglue_track_id": track_id, **link})
            assignments[frame1["frame_row"]] = target_assignment
            pair["mambaglue_track_ids"] = sorted(
                {item["mambaglue_track_id"] for item in linked}
            )
            pair["feature_track_links"] = linked
            for region in pair.get("regions", []):
                region["mambaglue_track_ids"] = sorted(
                    {
                        item["mambaglue_track_id"]
                        for item in linked
                        if item["source_region_id"] == region["region_id"]
                    }
                )
            source_pair = source_pairs[pair_id]
            pair["sift_comparison"] = {
                "matches": source_pair.get("matches"),
                "inliers": source_pair.get("inliers"),
                "median_flow_px": source_pair.get("median_flow_px"),
                "median_point_motion_px": source_pair.get("median_point_motion_px"),
                "angle_to_dominant_axis_deg": source_pair.get(
                    "angle_to_dominant_axis_deg"
                ),
                "stationary": source_pair.get("stationary"),
                "strong_transverse": source_pair.get("strong_transverse"),
            }
            if pair_id in candidate_pair_ids:
                if pytorch_matcher is None:
                    pytorch_matcher = MambaGlue(
                        features="superpoint",
                        flash=False,
                        mp=False,
                        depth_confidence=-1,
                        width_confidence=-1,
                        scan_backend="portable",
                    ).eval().cpu()
                pair["pytorch_onnx_comparison"] = pytorch_comparison(
                    pytorch_matcher, features0, features1, outputs
                )
            pair_results.append(pair)
            print(
                json.dumps(
                    {
                        "scene": source_scene["scene_id"],
                        "pair": pair_id,
                        "matches": pair.get("matches"),
                        "inliers": pair.get("inliers"),
                        "runtime_ms": pair.get("onnx_runtime_ms"),
                    }
                ),
                flush=True,
            )

        finalized = finalize_tracks(tracks, axis)
        selected_pair_ids = [
            item["pair_id"]
            for item in pair_results
            if item.get("stationary") or item.get("strong_transverse")
        ]
        selected_regions = sorted(
            {
                region["region_id"]
                for pair in pair_results
                if pair["pair_id"] in selected_pair_ids
                for region in pair.get("regions", [])
                if region.get("stationary") or region.get("strong_transverse")
            }
        )
        selected_tracks = sorted(
            {
                track_id
                for pair in pair_results
                if pair["pair_id"] in selected_pair_ids
                for region in pair.get("regions", [])
                if region["region_id"] in selected_regions
                for track_id in region["mambaglue_track_ids"]
            }
        )
        scene_results.append(
            {
                "scene_id": source_scene["scene_id"],
                "normal_motion_axis": source_scene["normal_motion_axis"],
                "frames": frames,
                "pair_results": pair_results,
                "tracks": finalized,
                "candidate_pair_ids": selected_pair_ids,
                "candidate_region_ids": selected_regions,
                "candidate_mambaglue_track_ids": selected_tracks,
                "track_count": len(finalized),
                "persistent_track_count_ge_3_observations": sum(
                    item["observation_count"] >= 3 for item in finalized
                ),
            }
        )

    comparisons = [
        pair["pytorch_onnx_comparison"]
        for scene in scene_results
        for pair in scene["pair_results"]
        if "pytorch_onnx_comparison" in pair
    ]
    return {
        "schema": "mambaglue_onnx_motion_analysis_v1",
        "source_analysis": str(args.source_analysis.resolve()),
        "source_analysis_sha256": sha256(args.source_analysis),
        "dataset_root": str(args.data_root.resolve()),
        "mambaglue": {
            "repository": "https://github.com/yuki-inaho/MambaGlue",
            "commit": args.repo_commit,
            "extractor": "SuperPoint PyTorch CPU",
            "extractor_max_keypoints": args.fixed_keypoints,
            "extractor_resize": args.resize,
            "matcher": "MambaGlue ONNX Runtime CPUExecutionProvider",
            "model_path": str(args.model.resolve()),
            "model_sha256": sha256(args.model),
            "model_bytes": args.model.stat().st_size,
            "model_metadata": onnx_matcher.session.get_modelmeta().custom_metadata_map,
            "onnx_inputs": onnx_matcher.input_names,
            "onnx_outputs": onnx_matcher.output_names,
            "torch_version": torch.__version__,
            "onnxruntime_version": ort.__version__,
            "execution_providers": onnx_matcher.session.get_providers(),
        },
        "classification_thresholds": {
            "minimum_pair_inliers": MIN_PAIR_INLIERS,
            "minimum_region_inliers": MIN_REGION_INLIERS,
            "perpendicular_angle_deg": PERPENDICULAR_ANGLE_DEG,
            "stationary_median_point_motion_px": STATIC_MOTION_PX,
            "strong_transverse_median_point_motion_px": STRONG_TRANSVERSE_MOTION_PX,
        },
        "pytorch_onnx_validation": {
            "candidate_pair_comparison_count": len(comparisons),
            "total_match_index_mismatches": sum(
                item["match_index_mismatch_count"] for item in comparisons
            ),
            "max_matching_score_abs_error": None
            if not comparisons
            else max(item["matching_score_max_abs_error"] for item in comparisons),
        },
        "scenes": scene_results,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--source-analysis", type=Path, required=True)
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--torch-home", type=Path, required=True)
    result.add_argument("--fixed-keypoints", type=int, default=256)
    result.add_argument("--resize", type=int, default=640)
    result.add_argument("--repo-commit", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    summary = {
        "output": str(args.output.resolve()),
        "pytorch_onnx_validation": report["pytorch_onnx_validation"],
        "scenes": [
            {
                "scene_id": scene["scene_id"],
                "tracks": scene["track_count"],
                "persistent_tracks": scene[
                    "persistent_track_count_ge_3_observations"
                ],
                "candidate_pairs": scene["candidate_pair_ids"],
                "candidate_regions": len(scene["candidate_region_ids"]),
                "candidate_tracks": len(scene["candidate_mambaglue_track_ids"]),
            }
            for scene in report["scenes"]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
