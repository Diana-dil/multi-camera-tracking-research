from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Повторная оценка сохранённых YOLO-детекций WILDTRACK "
            "при нескольких порогах IoU без повторного запуска модели."
        )
    )
    parser.add_argument(
        "--detections",
        type=Path,
        default=Path(
            "data/processed/wildtrack_C1_C6_yolo_50/detections.csv"
        ),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(
            "data/raw/WILDTRACK/annotations_positions"
        ),
    )
    parser.add_argument("--camera-a", default="C1")
    parser.add_argument("--camera-b", default="C6")
    parser.add_argument(
        "--thresholds",
        default="0.30,0.40,0.50,0.60,0.70",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/wildtrack_detection_iou_analysis_C1_C6"
        ),
    )
    return parser.parse_args()


def camera_to_view_number(camera_name: str) -> int:
    return int(camera_name.removeprefix("C")) - 1


def valid_view(view: dict) -> bool:
    x1 = float(view["xmin"])
    y1 = float(view["ymin"])
    x2 = float(view["xmax"])
    y2 = float(view["ymax"])

    return x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1


def load_gt(
    annotations_root: Path,
    frame_number: int,
    camera_name: str,
) -> pd.DataFrame:
    annotation_path = (
        annotations_root / f"{frame_number:08d}.json"
    )

    if not annotation_path.exists():
        raise FileNotFoundError(annotation_path)

    with annotation_path.open("r", encoding="utf-8") as file:
        annotations = json.load(file)

    view_number = camera_to_view_number(camera_name)
    rows: list[dict] = []

    for person in annotations:
        views = {
            int(view["viewNum"]): view
            for view in person.get("views", [])
        }

        view = views.get(view_number)

        if view is None or not valid_view(view):
            continue

        rows.append(
            {
                "person_id": int(person["personID"]),
                "x1": float(view["xmin"]),
                "y1": float(view["ymin"]),
                "x2": float(view["xmax"]),
                "y2": float(view["ymax"]),
            }
        )

    return pd.DataFrame(rows)


def pairwise_iou(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros(
            (len(boxes_a), len(boxes_b)),
            dtype=np.float64,
        )

    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = np.maximum(
        0.0,
        (a[..., 2] - a[..., 0])
        * (a[..., 3] - a[..., 1]),
    )
    area_b = np.maximum(
        0.0,
        (b[..., 2] - b[..., 0])
        * (b[..., 3] - b[..., 1]),
    )

    union = area_a + area_b - intersection

    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def optimal_assignment(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> list[tuple[int, int, float]]:
    if predictions.empty or ground_truth.empty:
        return []

    pred_boxes = predictions[
        ["x1", "y1", "x2", "y2"]
    ].to_numpy(dtype=np.float64)

    gt_boxes = ground_truth[
        ["x1", "y1", "x2", "y2"]
    ].to_numpy(dtype=np.float64)

    matrix = pairwise_iou(pred_boxes, gt_boxes)

    pred_indices, gt_indices = linear_sum_assignment(
        1.0 - matrix
    )

    return [
        (
            int(pred_index),
            int(gt_index),
            float(matrix[pred_index, gt_index]),
        )
        for pred_index, gt_index in zip(
            pred_indices,
            gt_indices,
        )
    ]


def main() -> None:
    args = parse_args()

    detections_path = args.detections.resolve()
    annotations_root = args.annotations.resolve()

    if not detections_path.exists():
        raise FileNotFoundError(detections_path)

    thresholds = [
        float(value.strip())
        for value in args.thresholds.split(",")
        if value.strip()
    ]

    detections = pd.read_csv(detections_path)
    detections["frame"] = detections["frame"].astype(int)

    required = {
        "frame",
        "camera_id",
        "x1",
        "y1",
        "x2",
        "y2",
    }

    missing = required - set(detections.columns)

    if missing:
        raise ValueError(
            f"В detections.csv отсутствуют поля: {sorted(missing)}"
        )

    cameras = [args.camera_a, args.camera_b]
    frame_numbers = sorted(detections["frame"].unique())

    metric_rows: list[dict] = []
    pair_rows: list[dict] = []
    iou_rows: list[dict] = []

    for frame_number in frame_numbers:
        gt_by_camera: dict[str, pd.DataFrame] = {}
        assignment_by_camera: dict[
            str,
            list[tuple[int, int, float]],
        ] = {}
        predictions_by_camera: dict[str, pd.DataFrame] = {}

        for camera in cameras:
            predictions = (
                detections[
                    (detections["frame"] == frame_number)
                    & (detections["camera_id"] == camera)
                ]
                .reset_index(drop=True)
            )

            ground_truth = load_gt(
                annotations_root=annotations_root,
                frame_number=int(frame_number),
                camera_name=camera,
            )

            assignments = optimal_assignment(
                predictions=predictions,
                ground_truth=ground_truth,
            )

            predictions_by_camera[camera] = predictions
            gt_by_camera[camera] = ground_truth
            assignment_by_camera[camera] = assignments

            for _, gt_index, iou in assignments:
                iou_rows.append(
                    {
                        "frame": int(frame_number),
                        "camera_id": camera,
                        "person_id": int(
                            ground_truth.iloc[gt_index]["person_id"]
                        ),
                        "assigned_iou": iou,
                    }
                )

            for threshold in thresholds:
                accepted = [
                    item
                    for item in assignments
                    if item[2] >= threshold
                ]

                tp = len(accepted)
                fp = len(predictions) - tp
                fn = len(ground_truth) - tp

                precision = (
                    tp / (tp + fp)
                    if tp + fp > 0
                    else 0.0
                )
                recall = (
                    tp / (tp + fn)
                    if tp + fn > 0
                    else 0.0
                )
                f1 = (
                    2.0 * precision * recall
                    / (precision + recall)
                    if precision + recall > 0
                    else 0.0
                )

                metric_rows.append(
                    {
                        "threshold_iou": threshold,
                        "frame": int(frame_number),
                        "camera_id": camera,
                        "gt_count": len(ground_truth),
                        "prediction_count": len(predictions),
                        "true_positive": tp,
                        "false_positive": fp,
                        "false_negative": fn,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    }
                )

        visible_a = set(
            gt_by_camera[args.camera_a]["person_id"].astype(int)
        )
        visible_b = set(
            gt_by_camera[args.camera_b]["person_id"].astype(int)
        )
        common_gt = visible_a & visible_b

        for threshold in thresholds:
            matched_ids: dict[str, set[int]] = {}

            for camera in cameras:
                ground_truth = gt_by_camera[camera]
                assignments = assignment_by_camera[camera]

                matched_ids[camera] = {
                    int(
                        ground_truth.iloc[gt_index]["person_id"]
                    )
                    for _, gt_index, iou in assignments
                    if iou >= threshold
                }

            both_detected = (
                common_gt
                & matched_ids[args.camera_a]
                & matched_ids[args.camera_b]
            )

            pair_rows.append(
                {
                    "threshold_iou": threshold,
                    "frame": int(frame_number),
                    "common_gt_pairs": len(common_gt),
                    "both_detected_pairs": len(both_detected),
                }
            )

    per_frame_metrics = pd.DataFrame(metric_rows)
    per_frame_pairs = pd.DataFrame(pair_rows)
    assigned_ious = pd.DataFrame(iou_rows)

    camera_summary = (
        per_frame_metrics.groupby(
            ["threshold_iou", "camera_id"],
            as_index=False,
        )
        .agg(
            gt_count=("gt_count", "sum"),
            prediction_count=("prediction_count", "sum"),
            true_positive=("true_positive", "sum"),
            false_positive=("false_positive", "sum"),
            false_negative=("false_negative", "sum"),
        )
    )

    camera_summary["precision"] = (
        camera_summary["true_positive"]
        / (
            camera_summary["true_positive"]
            + camera_summary["false_positive"]
        )
    )
    camera_summary["recall"] = (
        camera_summary["true_positive"]
        / (
            camera_summary["true_positive"]
            + camera_summary["false_negative"]
        )
    )
    camera_summary["f1"] = (
        2.0
        * camera_summary["precision"]
        * camera_summary["recall"]
        / (
            camera_summary["precision"]
            + camera_summary["recall"]
        )
    )

    total_summary = (
        per_frame_metrics.groupby(
            "threshold_iou",
            as_index=False,
        )
        .agg(
            gt_count=("gt_count", "sum"),
            prediction_count=("prediction_count", "sum"),
            true_positive=("true_positive", "sum"),
            false_positive=("false_positive", "sum"),
            false_negative=("false_negative", "sum"),
        )
    )

    total_summary["camera_id"] = "ALL"
    total_summary["precision"] = (
        total_summary["true_positive"]
        / (
            total_summary["true_positive"]
            + total_summary["false_positive"]
        )
    )
    total_summary["recall"] = (
        total_summary["true_positive"]
        / (
            total_summary["true_positive"]
            + total_summary["false_negative"]
        )
    )
    total_summary["f1"] = (
        2.0
        * total_summary["precision"]
        * total_summary["recall"]
        / (
            total_summary["precision"]
            + total_summary["recall"]
        )
    )

    detection_summary = pd.concat(
        [camera_summary, total_summary],
        ignore_index=True,
    ).sort_values(
        ["threshold_iou", "camera_id"]
    )

    pair_summary = (
        per_frame_pairs.groupby(
            "threshold_iou",
            as_index=False,
        )
        .agg(
            common_gt_pairs=("common_gt_pairs", "sum"),
            both_detected_pairs=("both_detected_pairs", "sum"),
        )
    )

    pair_summary["pair_detection_coverage"] = (
        pair_summary["both_detected_pairs"]
        / pair_summary["common_gt_pairs"]
    )

    iou_summary = (
        assigned_ious.groupby(
            "camera_id",
            as_index=False,
        )
        .agg(
            assigned_pairs=("assigned_iou", "count"),
            mean_iou=("assigned_iou", "mean"),
            median_iou=("assigned_iou", "median"),
            p10_iou=(
                "assigned_iou",
                lambda values: values.quantile(0.10),
            ),
            p25_iou=(
                "assigned_iou",
                lambda values: values.quantile(0.25),
            ),
            p75_iou=(
                "assigned_iou",
                lambda values: values.quantile(0.75),
            ),
            p90_iou=(
                "assigned_iou",
                lambda values: values.quantile(0.90),
            ),
        )
    )

    args.output.mkdir(parents=True, exist_ok=True)

    detection_summary.to_csv(
        args.output / "detection_summary_by_iou.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pair_summary.to_csv(
        args.output / "pair_coverage_by_iou.csv",
        index=False,
        encoding="utf-8-sig",
    )
    iou_summary.to_csv(
        args.output / "assigned_iou_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    per_frame_metrics.to_csv(
        args.output / "per_frame_detection_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Общие метрики детектора:")
    print(
        detection_summary[
            detection_summary["camera_id"] == "ALL"
        ].round(4).to_string(index=False)
    )
    print()
    print("Покрытие межкамерных GT-пар:")
    print(pair_summary.round(4).to_string(index=False))
    print()
    print("Распределение IoU назначенных пар:")
    print(iou_summary.round(4).to_string(index=False))
    print()
    print(f"Результаты: {args.output.resolve()}")


if __name__ == "__main__":
    main()
