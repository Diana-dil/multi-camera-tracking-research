from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


CAMERA_CALIBRATION_NAMES = {
    "C1": "CVLab1",
    "C2": "CVLab2",
    "C3": "CVLab3",
    "C4": "CVLab4",
    "C5": "IDIAP1",
    "C6": "IDIAP2",
    "C7": "IDIAP3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Диагностика YOLO-детекций WILDTRACK по расстоянию "
            "между проекциями нижних точек рамок и GT-позициями."
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
    parser.add_argument(
        "--calibrations",
        type=Path,
        default=Path(
            "data/raw/WILDTRACK/calibrations"
        ),
    )
    parser.add_argument("--camera-a", default="C1")
    parser.add_argument("--camera-b", default="C6")
    parser.add_argument(
        "--thresholds",
        default="0.25,0.50,0.75,1.00,1.50,2.00",
        help="Пороговые расстояния на земле, метры.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/wildtrack_detection_ground_analysis_C1_C6"
        ),
    )
    return parser.parse_args()


def read_xml_values(
    root: ET.Element,
    field_name: str,
) -> np.ndarray:
    node = root.find(field_name)

    if node is None:
        raise KeyError(
            f"Поле {field_name!r} не найдено в XML."
        )

    data_node = node.find("data")
    text = data_node.text if data_node is not None else node.text

    if not text:
        raise ValueError(
            f"Поле {field_name!r} не содержит значений."
        )

    return np.array(
        [float(value) for value in text.split()],
        dtype=np.float64,
    )


def load_calibration(
    calibrations_root: Path,
    camera_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    calibration_name = CAMERA_CALIBRATION_NAMES[camera_name]

    intrinsic_path = (
        calibrations_root
        / "intrinsic_zero"
        / f"intr_{calibration_name}.xml"
    )

    extrinsic_path = (
        calibrations_root
        / "extrinsic"
        / f"extr_{calibration_name}.xml"
    )

    if not intrinsic_path.exists():
        raise FileNotFoundError(intrinsic_path)

    if not extrinsic_path.exists():
        raise FileNotFoundError(extrinsic_path)

    intrinsic_root = ET.parse(intrinsic_path).getroot()
    extrinsic_root = ET.parse(extrinsic_path).getroot()

    camera_matrix = read_xml_values(
        intrinsic_root,
        "camera_matrix",
    ).reshape(3, 3)

    rvec = read_xml_values(
        extrinsic_root,
        "rvec",
    ).reshape(3, 1)

    tvec = read_xml_values(
        extrinsic_root,
        "tvec",
    ).reshape(3, 1)

    return camera_matrix, rvec, tvec


def image_point_to_ground(
    u: float,
    v: float,
    camera_matrix: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    camera_center_world = (
        -rotation_matrix.T @ tvec
    ).reshape(3)

    pixel = np.array(
        [u, v, 1.0],
        dtype=np.float64,
    )

    ray_camera = np.linalg.inv(camera_matrix) @ pixel
    ray_world = rotation_matrix.T @ ray_camera

    if abs(ray_world[2]) < 1e-10:
        raise ValueError(
            "Луч почти параллелен плоскости земли."
        )

    scale = -camera_center_world[2] / ray_world[2]

    world_point_cm = (
        camera_center_world
        + scale * ray_world
    )

    return world_point_cm[:2] / 100.0


def position_id_to_world_m(
    position_id: int,
) -> np.ndarray:
    grid_x = position_id % 480
    grid_y = position_id // 480

    return np.array(
        [
            -3.0 + 0.025 * grid_x,
            -9.0 + 0.025 * grid_y,
        ],
        dtype=np.float64,
    )


def camera_to_view_number(
    camera_name: str,
) -> int:
    return int(camera_name.removeprefix("C")) - 1


def valid_view(view: dict) -> bool:
    x1 = float(view["xmin"])
    y1 = float(view["ymin"])
    x2 = float(view["xmax"])
    y2 = float(view["ymax"])

    return (
        x1 >= 0
        and y1 >= 0
        and x2 > x1
        and y2 > y1
    )


def load_gt(
    annotations_root: Path,
    frame_number: int,
    camera_name: str,
) -> pd.DataFrame:
    annotation_path = (
        annotations_root
        / f"{frame_number:08d}.json"
    )

    if not annotation_path.exists():
        raise FileNotFoundError(annotation_path)

    with annotation_path.open(
        "r",
        encoding="utf-8",
    ) as file:
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

        position_id = int(person["positionID"])
        point = position_id_to_world_m(position_id)

        rows.append(
            {
                "person_id": int(person["personID"]),
                "position_id": position_id,
                "gt_x_m": float(point[0]),
                "gt_y_m": float(point[1]),
            }
        )

    return pd.DataFrame(rows)


def project_detections(
    detections: pd.DataFrame,
    calibrations_root: Path,
) -> pd.DataFrame:
    result = detections.copy()

    calibrations = {
        camera: load_calibration(
            calibrations_root,
            camera,
        )
        for camera in result["camera_id"].unique()
    }

    xs: list[float] = []
    ys: list[float] = []

    for row in result.itertuples():
        camera_matrix, rvec, tvec = calibrations[
            str(row.camera_id)
        ]

        point = image_point_to_ground(
            u=float(row.foot_u),
            v=float(row.foot_v),
            camera_matrix=camera_matrix,
            rvec=rvec,
            tvec=tvec,
        )

        xs.append(float(point[0]))
        ys.append(float(point[1]))

    result["pred_x_m"] = xs
    result["pred_y_m"] = ys

    return result


def optimal_assignment(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> list[tuple[int, int, float]]:
    if predictions.empty or ground_truth.empty:
        return []

    pred_xy = predictions[
        ["pred_x_m", "pred_y_m"]
    ].to_numpy(dtype=np.float64)

    gt_xy = ground_truth[
        ["gt_x_m", "gt_y_m"]
    ].to_numpy(dtype=np.float64)

    distance_matrix = np.linalg.norm(
        pred_xy[:, None, :]
        - gt_xy[None, :, :],
        axis=2,
    )

    pred_indices, gt_indices = (
        linear_sum_assignment(distance_matrix)
    )

    return [
        (
            int(pred_index),
            int(gt_index),
            float(
                distance_matrix[
                    pred_index,
                    gt_index,
                ]
            ),
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
    calibrations_root = args.calibrations.resolve()

    if not detections_path.exists():
        raise FileNotFoundError(detections_path)

    thresholds = [
        float(value.strip())
        for value in args.thresholds.split(",")
        if value.strip()
    ]

    detections = pd.read_csv(detections_path)

    required_columns = {
        "frame",
        "camera_id",
        "foot_u",
        "foot_v",
    }

    missing = (
        required_columns
        - set(detections.columns)
    )

    if missing:
        raise ValueError(
            "В detections.csv отсутствуют поля: "
            f"{sorted(missing)}"
        )

    detections["frame"] = (
        detections["frame"].astype(int)
    )

    cameras = [
        args.camera_a,
        args.camera_b,
    ]

    detections = detections[
        detections["camera_id"].isin(cameras)
    ].reset_index(drop=True)

    detections = project_detections(
        detections=detections,
        calibrations_root=calibrations_root,
    )

    frame_numbers = sorted(
        detections["frame"].unique()
    )

    metric_rows: list[dict] = []
    pair_rows: list[dict] = []
    assignment_rows: list[dict] = []

    for frame_number in frame_numbers:
        gt_by_camera: dict[str, pd.DataFrame] = {}
        assignments_by_camera: dict[
            str,
            list[tuple[int, int, float]],
        ] = {}

        for camera in cameras:
            predictions = (
                detections[
                    (detections["frame"] == frame_number)
                    & (
                        detections["camera_id"]
                        == camera
                    )
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

            gt_by_camera[camera] = ground_truth
            assignments_by_camera[camera] = assignments

            for (
                prediction_index,
                gt_index,
                distance_m,
            ) in assignments:
                assignment_rows.append(
                    {
                        "frame": int(frame_number),
                        "camera_id": camera,
                        "prediction_index": (
                            prediction_index
                        ),
                        "person_id": int(
                            ground_truth.iloc[
                                gt_index
                            ]["person_id"]
                        ),
                        "ground_distance_m": (
                            distance_m
                        ),
                    }
                )

            for threshold in thresholds:
                accepted = [
                    assignment
                    for assignment in assignments
                    if assignment[2] <= threshold
                ]

                true_positive = len(accepted)
                false_positive = (
                    len(predictions)
                    - true_positive
                )
                false_negative = (
                    len(ground_truth)
                    - true_positive
                )

                precision = (
                    true_positive
                    / (
                        true_positive
                        + false_positive
                    )
                    if (
                        true_positive
                        + false_positive
                    ) > 0
                    else 0.0
                )

                recall = (
                    true_positive
                    / (
                        true_positive
                        + false_negative
                    )
                    if (
                        true_positive
                        + false_negative
                    ) > 0
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
                        "threshold_m": threshold,
                        "frame": int(frame_number),
                        "camera_id": camera,
                        "gt_count": len(ground_truth),
                        "prediction_count": (
                            len(predictions)
                        ),
                        "true_positive": (
                            true_positive
                        ),
                        "false_positive": (
                            false_positive
                        ),
                        "false_negative": (
                            false_negative
                        ),
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    }
                )

        visible_a = set(
            gt_by_camera[
                args.camera_a
            ]["person_id"].astype(int)
        )

        visible_b = set(
            gt_by_camera[
                args.camera_b
            ]["person_id"].astype(int)
        )

        common_ids = (
            visible_a & visible_b
        )

        for threshold in thresholds:
            matched_ids: dict[str, set[int]] = {}

            for camera in cameras:
                ground_truth = gt_by_camera[camera]
                assignments = (
                    assignments_by_camera[camera]
                )

                matched_ids[camera] = {
                    int(
                        ground_truth.iloc[
                            gt_index
                        ]["person_id"]
                    )
                    for (
                        _,
                        gt_index,
                        distance_m,
                    ) in assignments
                    if distance_m <= threshold
                }

            both_detected = (
                common_ids
                & matched_ids[args.camera_a]
                & matched_ids[args.camera_b]
            )

            pair_rows.append(
                {
                    "threshold_m": threshold,
                    "frame": int(frame_number),
                    "common_gt_pairs": (
                        len(common_ids)
                    ),
                    "both_detected_pairs": (
                        len(both_detected)
                    ),
                }
            )

    per_frame_metrics = pd.DataFrame(
        metric_rows
    )

    per_frame_pairs = pd.DataFrame(
        pair_rows
    )

    assignments = pd.DataFrame(
        assignment_rows
    )

    camera_summary = (
        per_frame_metrics.groupby(
            ["threshold_m", "camera_id"],
            as_index=False,
        )
        .agg(
            gt_count=("gt_count", "sum"),
            prediction_count=(
                "prediction_count",
                "sum",
            ),
            true_positive=(
                "true_positive",
                "sum",
            ),
            false_positive=(
                "false_positive",
                "sum",
            ),
            false_negative=(
                "false_negative",
                "sum",
            ),
        )
    )

    camera_summary["precision"] = (
        camera_summary["true_positive"]
        / (
            camera_summary["true_positive"]
            + camera_summary[
                "false_positive"
            ]
        )
    )

    camera_summary["recall"] = (
        camera_summary["true_positive"]
        / (
            camera_summary["true_positive"]
            + camera_summary[
                "false_negative"
            ]
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
            "threshold_m",
            as_index=False,
        )
        .agg(
            gt_count=("gt_count", "sum"),
            prediction_count=(
                "prediction_count",
                "sum",
            ),
            true_positive=(
                "true_positive",
                "sum",
            ),
            false_positive=(
                "false_positive",
                "sum",
            ),
            false_negative=(
                "false_negative",
                "sum",
            ),
        )
    )

    total_summary["camera_id"] = "ALL"

    total_summary["precision"] = (
        total_summary["true_positive"]
        / (
            total_summary["true_positive"]
            + total_summary[
                "false_positive"
            ]
        )
    )

    total_summary["recall"] = (
        total_summary["true_positive"]
        / (
            total_summary["true_positive"]
            + total_summary[
                "false_negative"
            ]
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
        [
            camera_summary,
            total_summary,
        ],
        ignore_index=True,
    ).sort_values(
        [
            "threshold_m",
            "camera_id",
        ]
    )

    pair_summary = (
        per_frame_pairs.groupby(
            "threshold_m",
            as_index=False,
        )
        .agg(
            common_gt_pairs=(
                "common_gt_pairs",
                "sum",
            ),
            both_detected_pairs=(
                "both_detected_pairs",
                "sum",
            ),
        )
    )

    pair_summary[
        "pair_detection_coverage"
    ] = (
        pair_summary["both_detected_pairs"]
        / pair_summary["common_gt_pairs"]
    )

    assignment_summary = (
        assignments.groupby(
            "camera_id",
            as_index=False,
        )
        .agg(
            assigned_pairs=(
                "ground_distance_m",
                "count",
            ),
            mean_distance_m=(
                "ground_distance_m",
                "mean",
            ),
            median_distance_m=(
                "ground_distance_m",
                "median",
            ),
            p10_distance_m=(
                "ground_distance_m",
                lambda values: values.quantile(
                    0.10
                ),
            ),
            p25_distance_m=(
                "ground_distance_m",
                lambda values: values.quantile(
                    0.25
                ),
            ),
            p75_distance_m=(
                "ground_distance_m",
                lambda values: values.quantile(
                    0.75
                ),
            ),
            p90_distance_m=(
                "ground_distance_m",
                lambda values: values.quantile(
                    0.90
                ),
            ),
        )
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    detection_summary.to_csv(
        args.output
        / "detection_summary_by_ground_distance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pair_summary.to_csv(
        args.output
        / "pair_coverage_by_ground_distance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    assignment_summary.to_csv(
        args.output
        / "assigned_ground_distance_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    assignments.to_csv(
        args.output
        / "assigned_ground_distances.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Общие метрики:")
    print(
        detection_summary[
            detection_summary["camera_id"]
            == "ALL"
        ].round(4).to_string(index=False)
    )

    print()
    print("Покрытие межкамерных пар:")
    print(
        pair_summary.round(4).to_string(
            index=False
        )
    )

    print()
    print("Распределение расстояний:")
    print(
        assignment_summary.round(4).to_string(
            index=False
        )
    )

    print()
    print(
        "Результаты:",
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
