from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REQUIRED_COLUMNS = {"frame_index", "track_id", "x1", "y1", "x2", "y2"}


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float | int]
    matches: pd.DataFrame
    per_track: pd.DataFrame


def _validate(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    clean = frame.copy()
    clean["frame_index"] = clean["frame_index"].astype(int)
    clean["track_id"] = clean["track_id"].astype(int)
    for column in ["x1", "y1", "x2", "y2"]:
        clean[column] = pd.to_numeric(clean[column], errors="raise")
    invalid = (clean["x2"] <= clean["x1"]) | (clean["y2"] <= clean["y1"])
    if invalid.any():
        raise ValueError(f"{name} contains {int(invalid.sum())} invalid bounding boxes")
    return clean


def box_iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """Return pairwise IoU matrix for xyxy boxes."""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float64)

    gt = gt_boxes[:, None, :]
    pred = pred_boxes[None, :, :]
    inter_x1 = np.maximum(gt[..., 0], pred[..., 0])
    inter_y1 = np.maximum(gt[..., 1], pred[..., 1])
    inter_x2 = np.minimum(gt[..., 2], pred[..., 2])
    inter_y2 = np.minimum(gt[..., 3], pred[..., 3])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    gt_area = (gt[..., 2] - gt[..., 0]) * (gt[..., 3] - gt[..., 1])
    pred_area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
    union = gt_area + pred_area - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0,
    )


def _frame_matches(
    gt_frame: pd.DataFrame,
    pred_frame: pd.DataFrame,
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    if gt_frame.empty or pred_frame.empty:
        return []

    gt_boxes = gt_frame[["x1", "y1", "x2", "y2"]].to_numpy(float)
    pred_boxes = pred_frame[["x1", "y1", "x2", "y2"]].to_numpy(float)
    ious = box_iou_matrix(gt_boxes, pred_boxes)
    cost = 1.0 - ious
    cost[ious < iou_threshold] = 1e6
    gt_indices, pred_indices = linear_sum_assignment(cost)

    matches: list[tuple[int, int, float]] = []
    for gt_idx, pred_idx in zip(gt_indices, pred_indices, strict=True):
        iou = float(ious[gt_idx, pred_idx])
        if iou >= iou_threshold:
            matches.append((int(gt_idx), int(pred_idx), iou))
    return matches


def evaluate_tracking(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    iou_threshold: float = 0.5,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> EvaluationResult:
    """Evaluate tracking with transparent MOT-style metrics.

    This evaluator is intended for local research checks. For publication-grade HOTA,
    run the exported MOTChallenge files through the official TrackEval package.
    """
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")

    gt = _validate(ground_truth, "ground truth")
    pred = _validate(predictions, "predictions")

    if start_frame is not None:
        gt = gt[gt.frame_index >= start_frame]
        pred = pred[pred.frame_index >= start_frame]
    if end_frame is not None:
        gt = gt[gt.frame_index <= end_frame]
        pred = pred[pred.frame_index <= end_frame]

    gt = gt.sort_values(["frame_index", "track_id"]).reset_index(drop=True)
    pred = pred.sort_values(["frame_index", "track_id"]).reset_index(drop=True)

    frames = sorted(set(gt.frame_index.unique()) | set(pred.frame_index.unique()))
    match_rows: list[dict] = []
    matched_pairs: list[tuple[int, int]] = []
    previous_prediction_by_gt: dict[int, int] = {}
    id_switches = 0
    false_positives = 0
    false_negatives = 0
    matched_ious: list[float] = []
    matched_state_by_gt: dict[int, dict[int, bool]] = {}

    for frame_index in frames:
        gt_frame = gt[gt.frame_index == frame_index].reset_index(drop=True)
        pred_frame = pred[pred.frame_index == frame_index].reset_index(drop=True)
        matches = _frame_matches(gt_frame, pred_frame, iou_threshold)
        matched_gt_indices = {item[0] for item in matches}
        matched_pred_indices = {item[1] for item in matches}

        false_negatives += len(gt_frame) - len(matched_gt_indices)
        false_positives += len(pred_frame) - len(matched_pred_indices)

        matched_gt_ids: set[int] = set()
        for gt_idx, pred_idx, iou in matches:
            gt_id = int(gt_frame.iloc[gt_idx].track_id)
            pred_id = int(pred_frame.iloc[pred_idx].track_id)
            previous_pred_id = previous_prediction_by_gt.get(gt_id)
            is_switch = previous_pred_id is not None and previous_pred_id != pred_id
            if is_switch:
                id_switches += 1
            previous_prediction_by_gt[gt_id] = pred_id
            matched_gt_ids.add(gt_id)
            matched_pairs.append((gt_id, pred_id))
            matched_ious.append(iou)
            match_rows.append(
                {
                    "frame_index": int(frame_index),
                    "gt_id": gt_id,
                    "pred_id": pred_id,
                    "iou": round(iou, 6),
                    "id_switch": bool(is_switch),
                }
            )

        for gt_id in gt_frame.track_id.astype(int):
            matched_state_by_gt.setdefault(gt_id, {})[int(frame_index)] = gt_id in matched_gt_ids

    total_gt = int(len(gt))
    total_pred = int(len(pred))
    true_positives = len(matched_pairs)
    precision = true_positives / total_pred if total_pred else 0.0
    recall = true_positives / total_gt if total_gt else 0.0
    mota = 1.0 - (false_negatives + false_positives + id_switches) / total_gt if total_gt else 0.0
    motp_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    gt_ids = sorted(gt.track_id.unique().astype(int).tolist())
    pred_ids = sorted(pred.track_id.unique().astype(int).tolist())
    gt_index = {track_id: idx for idx, track_id in enumerate(gt_ids)}
    pred_index = {track_id: idx for idx, track_id in enumerate(pred_ids)}
    identity_counts = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
    for gt_id, pred_id in matched_pairs:
        identity_counts[gt_index[gt_id], pred_index[pred_id]] += 1

    idtp = 0
    if identity_counts.size:
        row_ind, col_ind = linear_sum_assignment(-identity_counts)
        idtp = int(identity_counts[row_ind, col_ind].sum())
    idfn = total_gt - idtp
    idfp = total_pred - idtp
    id_precision = idtp / (idtp + idfp) if idtp + idfp else 0.0
    id_recall = idtp / (idtp + idfn) if idtp + idfn else 0.0
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if 2 * idtp + idfp + idfn else 0.0

    per_track_rows: list[dict] = []
    fragmentations = 0
    mostly_tracked = 0
    mostly_lost = 0
    for gt_id, group in gt.groupby("track_id"):
        visible_frames = sorted(group.frame_index.astype(int).tolist())
        states = [matched_state_by_gt.get(int(gt_id), {}).get(frame, False) for frame in visible_frames]
        matched_count = int(sum(states))
        coverage = matched_count / len(states) if states else 0.0
        segments = 0
        previous = False
        for state in states:
            if state and not previous:
                segments += 1
            previous = state
        track_fragmentations = max(0, segments - 1)
        fragmentations += track_fragmentations
        mostly_tracked += int(coverage >= 0.8)
        mostly_lost += int(coverage <= 0.2)
        per_track_rows.append(
            {
                "gt_id": int(gt_id),
                "gt_detections": len(states),
                "matched_detections": matched_count,
                "coverage": round(coverage, 6),
                "fragments": track_fragmentations,
            }
        )

    metrics: dict[str, float | int] = {
        "iou_threshold": float(iou_threshold),
        "frames_evaluated": len(frames),
        "gt_detections": total_gt,
        "pred_detections": total_pred,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "mota": round(mota, 6),
        "motp_iou": round(motp_iou, 6),
        "id_switches": id_switches,
        "fragmentations": fragmentations,
        "mostly_tracked": mostly_tracked,
        "mostly_lost": mostly_lost,
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
        "id_precision": round(id_precision, 6),
        "id_recall": round(id_recall, 6),
        "idf1": round(idf1, 6),
    }
    return EvaluationResult(
        metrics=metrics,
        matches=pd.DataFrame(
            match_rows,
            columns=["frame_index", "gt_id", "pred_id", "iou", "id_switch"],
        ),
        per_track=pd.DataFrame(
            per_track_rows,
            columns=["gt_id", "gt_detections", "matched_detections", "coverage", "fragments"],
        ),
    )


def write_motchallenge_file(frame: pd.DataFrame, output: str | Path) -> Path:
    """Write observations as MOTChallenge: frame,id,x,y,w,h,conf,-1,-1,-1."""
    data = _validate(frame, "tracking data")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    confidence = data["confidence"] if "confidence" in data.columns else pd.Series(1.0, index=data.index)
    mot = pd.DataFrame(
        {
            "frame": data.frame_index.astype(int) + 1,
            "id": data.track_id.astype(int),
            "x": data.x1.astype(float),
            "y": data.y1.astype(float),
            "w": data.x2.astype(float) - data.x1.astype(float),
            "h": data.y2.astype(float) - data.y1.astype(float),
            "conf": confidence.astype(float),
            "x3d": -1,
            "y3d": -1,
            "z3d": -1,
        }
    )
    mot.to_csv(output_path, index=False, header=False)
    return output_path
