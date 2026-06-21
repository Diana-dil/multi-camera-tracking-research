from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


INF_COST = 1_000_000.0


def parse_float_list(value: str) -> list[float]:
    return [
        float(item.strip())
        for item in value.split(",")
        if item.strip()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Глобальное взаимно-однозначное межкамерное "
            "сопоставление WILDTRACK венгерским алгоритмом."
        )
    )

    parser.add_argument(
        "--features-dir",
        type=Path,
        required=True,
        help=(
            "Папка с detections_with_features.csv "
            "и embeddings.npy."
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
    parser.add_argument("--camera-b", default="C3")

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.5,
        help=(
            "Доля первых временных точек для настройки "
            "параметров. Остальные используются как test."
        ),
    )

    parser.add_argument(
        "--geometry-thresholds",
        default="0.50,0.75,1.00,1.50,2.00",
        help="Геометрические пороги в метрах.",
    )

    parser.add_argument(
        "--alphas",
        default="0.25,0.50,0.75",
        help=(
            "Вес appearance-cost в объединённой стоимости. "
            "Вес геометрии равен 1-alpha."
        ),
    )

    parser.add_argument(
        "--acceptance-costs",
        default="0.20,0.30,0.40,0.50,0.60,0.70,0.80",
        help=(
            "Максимальная стоимость, при которой выгоднее "
            "связать пару, чем оставить оба объекта unmatched."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/wildtrack_hungarian_C1_C3"
        ),
    )

    return parser.parse_args()


def camera_to_view_number(camera_name: str) -> int:
    return int(camera_name.removeprefix("C")) - 1


def is_valid_view(view: dict) -> bool:
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


def load_common_gt_ids(
    annotations_root: Path,
    frame_number: int,
    camera_a: str,
    camera_b: str,
) -> set[int]:
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

    visible: dict[str, set[int]] = {
        camera_a: set(),
        camera_b: set(),
    }

    view_numbers = {
        camera_a: camera_to_view_number(camera_a),
        camera_b: camera_to_view_number(camera_b),
    }

    for person in annotations:
        person_id = int(person["personID"])

        views = {
            int(view["viewNum"]): view
            for view in person.get("views", [])
        }

        for camera in (camera_a, camera_b):
            view = views.get(view_numbers[camera])

            if (
                view is not None
                and is_valid_view(view)
            ):
                visible[camera].add(person_id)

    return visible[camera_a] & visible[camera_b]


def build_real_cost_matrix(
    query_group: pd.DataFrame,
    gallery_group: pd.DataFrame,
    embeddings: np.ndarray,
    method: str,
    geometry_threshold_m: float | None,
    alpha: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query_indices = (
        query_group["embedding_index"]
        .astype(int)
        .to_numpy()
    )

    gallery_indices = (
        gallery_group["embedding_index"]
        .astype(int)
        .to_numpy()
    )

    query_embeddings = embeddings[query_indices]
    gallery_embeddings = embeddings[gallery_indices]

    similarities = (
        query_embeddings
        @ gallery_embeddings.T
    )

    appearance_cost = 1.0 - similarities

    query_xy = query_group[
        ["ground_x_m", "ground_y_m"]
    ].to_numpy(dtype=np.float64)

    gallery_xy = gallery_group[
        ["ground_x_m", "ground_y_m"]
    ].to_numpy(dtype=np.float64)

    distances = np.linalg.norm(
        query_xy[:, None, :]
        - gallery_xy[None, :, :],
        axis=2,
    )

    if method == "appearance":
        real_cost = appearance_cost.copy()

    elif method == "geometry":
        if geometry_threshold_m is None:
            raise ValueError(
                "Для geometry нужен geometry_threshold_m."
            )

        real_cost = (
            distances / geometry_threshold_m
        )

        real_cost[
            distances > geometry_threshold_m
        ] = INF_COST

    elif method == "fused":
        if (
            geometry_threshold_m is None
            or alpha is None
        ):
            raise ValueError(
                "Для fused нужны threshold и alpha."
            )

        geometry_cost = (
            distances / geometry_threshold_m
        )

        real_cost = (
            alpha * appearance_cost
            + (1.0 - alpha) * geometry_cost
        )

        real_cost[
            distances > geometry_threshold_m
        ] = INF_COST

    else:
        raise ValueError(
            f"Неизвестный method: {method}"
        )

    return real_cost, similarities, distances


def solve_with_unmatched(
    real_cost: np.ndarray,
    acceptance_cost: float,
) -> list[tuple[int, int, float]]:
    """
    Венгерское назначение с dummy-узлами.

    Реальная пара выбирается только тогда, когда её стоимость
    выгоднее, чем оставить оба объекта unmatched.
    """
    query_count, gallery_count = real_cost.shape

    if query_count == 0 or gallery_count == 0:
        return []

    total_size = query_count + gallery_count

    augmented = np.full(
        (total_size, total_size),
        INF_COST,
        dtype=np.float64,
    )

    augmented[
        :query_count,
        :gallery_count,
    ] = real_cost

    dummy_cost = acceptance_cost / 2.0

    # Каждая query-детекция может остаться unmatched.
    for query_index in range(query_count):
        augmented[
            query_index,
            gallery_count + query_index,
        ] = dummy_cost

    # Каждая gallery-детекция может остаться unmatched.
    for gallery_index in range(gallery_count):
        augmented[
            query_count + gallery_index,
            gallery_index,
        ] = dummy_cost

    # Dummy ↔ dummy заполняет оставшиеся назначения.
    augmented[
        query_count:,
        gallery_count:,
    ] = 0.0

    row_indices, column_indices = (
        linear_sum_assignment(augmented)
    )

    accepted: list[tuple[int, int, float]] = []

    for row_index, column_index in zip(
        row_indices,
        column_indices,
    ):
        if (
            row_index < query_count
            and column_index < gallery_count
        ):
            cost = float(
                real_cost[
                    row_index,
                    column_index,
                ]
            )

            if (
                cost < INF_COST / 2.0
                and cost <= acceptance_cost
            ):
                accepted.append(
                    (
                        int(row_index),
                        int(column_index),
                        cost,
                    )
                )

    return accepted


def evaluate_frames(
    detections: pd.DataFrame,
    embeddings: np.ndarray,
    annotations_root: Path,
    frames: list[int],
    camera_a: str,
    camera_b: str,
    method: str,
    geometry_threshold_m: float | None,
    alpha: float | None,
    acceptance_cost: float,
    keep_pair_rows: bool = False,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    frame_rows: list[dict] = []
    pair_rows: list[dict] = []

    total_common_gt = 0
    total_detectable_gt = 0
    total_predicted_pairs = 0
    total_true_positive = 0
    total_false_positive = 0

    for frame_number in frames:
        common_gt_ids = load_common_gt_ids(
            annotations_root=annotations_root,
            frame_number=frame_number,
            camera_a=camera_a,
            camera_b=camera_b,
        )

        frame_detections = detections[
            detections["frame"] == frame_number
        ]

        query_group = (
            frame_detections[
                frame_detections["camera_id"]
                == camera_a
            ]
            .reset_index(drop=True)
        )

        gallery_group = (
            frame_detections[
                frame_detections["camera_id"]
                == camera_b
            ]
            .reset_index(drop=True)
        )

        matched_query_ids = set(
            query_group.loc[
                (
                    query_group["matched_gt"] == 1
                )
                & (
                    query_group["gt_person_id"] >= 0
                ),
                "gt_person_id",
            ].astype(int)
        )

        matched_gallery_ids = set(
            gallery_group.loc[
                (
                    gallery_group["matched_gt"] == 1
                )
                & (
                    gallery_group["gt_person_id"] >= 0
                ),
                "gt_person_id",
            ].astype(int)
        )

        detectable_gt_ids = (
            common_gt_ids
            & matched_query_ids
            & matched_gallery_ids
        )

        if (
            query_group.empty
            or gallery_group.empty
        ):
            accepted_pairs: list[
                tuple[int, int, float]
            ] = []
            similarities = np.empty(
                (
                    len(query_group),
                    len(gallery_group),
                )
            )
            distances = np.empty_like(
                similarities
            )
        else:
            (
                real_cost,
                similarities,
                distances,
            ) = build_real_cost_matrix(
                query_group=query_group,
                gallery_group=gallery_group,
                embeddings=embeddings,
                method=method,
                geometry_threshold_m=(
                    geometry_threshold_m
                ),
                alpha=alpha,
            )

            accepted_pairs = solve_with_unmatched(
                real_cost=real_cost,
                acceptance_cost=acceptance_cost,
            )

        # Сначала дешёвые пары. Только одна пара может дать TP
        # для одного и того же global ID.
        accepted_pairs = sorted(
            accepted_pairs,
            key=lambda item: item[2],
        )

        correctly_matched_ids: set[int] = set()
        frame_false_positive = 0

        for (
            query_position,
            gallery_position,
            pair_cost,
        ) in accepted_pairs:
            query_row = query_group.iloc[
                query_position
            ]
            gallery_row = gallery_group.iloc[
                gallery_position
            ]

            query_gt_id = int(
                query_row["gt_person_id"]
            )
            gallery_gt_id = int(
                gallery_row["gt_person_id"]
            )

            is_true_pair = (
                int(query_row["matched_gt"]) == 1
                and int(
                    gallery_row["matched_gt"]
                ) == 1
                and query_gt_id >= 0
                and query_gt_id == gallery_gt_id
                and query_gt_id in common_gt_ids
                and query_gt_id
                not in correctly_matched_ids
            )

            if is_true_pair:
                correctly_matched_ids.add(
                    query_gt_id
                )
            else:
                frame_false_positive += 1

            if keep_pair_rows:
                pair_rows.append(
                    {
                        "frame": frame_number,
                        "query_camera": camera_a,
                        "gallery_camera": camera_b,
                        "query_detection_id": int(
                            query_row["detection_id"]
                        ),
                        "gallery_detection_id": int(
                            gallery_row["detection_id"]
                        ),
                        "query_gt_person_id": (
                            query_gt_id
                        ),
                        "gallery_gt_person_id": (
                            gallery_gt_id
                        ),
                        "pair_cost": pair_cost,
                        "cosine_similarity": float(
                            similarities[
                                query_position,
                                gallery_position,
                            ]
                        ),
                        "ground_distance_m": float(
                            distances[
                                query_position,
                                gallery_position,
                            ]
                        ),
                        "is_true_pair": int(
                            is_true_pair
                        ),
                    }
                )

        frame_true_positive = len(
            correctly_matched_ids
        )

        frame_common_gt = len(
            common_gt_ids
        )

        frame_detectable_gt = len(
            detectable_gt_ids
        )

        frame_predicted_pairs = len(
            accepted_pairs
        )

        frame_false_negative = (
            frame_common_gt
            - frame_true_positive
        )

        frame_precision = (
            frame_true_positive
            / frame_predicted_pairs
            if frame_predicted_pairs > 0
            else 0.0
        )

        frame_recall = (
            frame_true_positive
            / frame_common_gt
            if frame_common_gt > 0
            else 0.0
        )

        frame_f1 = (
            2.0
            * frame_precision
            * frame_recall
            / (
                frame_precision
                + frame_recall
            )
            if (
                frame_precision
                + frame_recall
            ) > 0
            else 0.0
        )

        frame_conditional_accuracy = (
            frame_true_positive
            / frame_detectable_gt
            if frame_detectable_gt > 0
            else 0.0
        )

        frame_rows.append(
            {
                "frame": frame_number,
                "common_gt_ids": frame_common_gt,
                "detectable_gt_ids": (
                    frame_detectable_gt
                ),
                "predicted_pairs": (
                    frame_predicted_pairs
                ),
                "true_positive": (
                    frame_true_positive
                ),
                "false_positive": (
                    frame_false_positive
                ),
                "false_negative": (
                    frame_false_negative
                ),
                "precision": frame_precision,
                "recall": frame_recall,
                "f1": frame_f1,
                "conditional_accuracy": (
                    frame_conditional_accuracy
                ),
            }
        )

        total_common_gt += frame_common_gt
        total_detectable_gt += (
            frame_detectable_gt
        )
        total_predicted_pairs += (
            frame_predicted_pairs
        )
        total_true_positive += (
            frame_true_positive
        )
        total_false_positive += (
            frame_false_positive
        )

    total_false_negative = (
        total_common_gt
        - total_true_positive
    )

    precision = (
        total_true_positive
        / total_predicted_pairs
        if total_predicted_pairs > 0
        else 0.0
    )

    recall = (
        total_true_positive
        / total_common_gt
        if total_common_gt > 0
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    conditional_accuracy = (
        total_true_positive
        / total_detectable_gt
        if total_detectable_gt > 0
        else 0.0
    )

    pair_detection_coverage = (
        total_detectable_gt
        / total_common_gt
        if total_common_gt > 0
        else 0.0
    )

    metrics = {
        "method": method,
        "geometry_threshold_m": (
            geometry_threshold_m
        ),
        "alpha": alpha,
        "acceptance_cost": acceptance_cost,
        "frames": len(frames),
        "common_gt_ids": total_common_gt,
        "detectable_gt_ids": (
            total_detectable_gt
        ),
        "pair_detection_coverage": (
            pair_detection_coverage
        ),
        "predicted_pairs": (
            total_predicted_pairs
        ),
        "true_positive": (
            total_true_positive
        ),
        "false_positive": (
            total_false_positive
        ),
        "false_negative": (
            total_false_negative
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "conditional_accuracy": (
            conditional_accuracy
        ),
    }

    return (
        metrics,
        pd.DataFrame(frame_rows),
        pd.DataFrame(pair_rows),
    )


def choose_best(
    table: pd.DataFrame,
) -> pd.Series:
    ordered = table.sort_values(
        [
            "f1",
            "conditional_accuracy",
            "precision",
            "recall",
        ],
        ascending=False,
    )

    return ordered.iloc[0]


def main() -> None:
    args = parse_args()

    features_dir = args.features_dir.resolve()

    detections_path = (
        features_dir
        / "detections_with_features.csv"
    )

    embeddings_path = (
        features_dir
        / "embeddings.npy"
    )

    if not detections_path.exists():
        raise FileNotFoundError(
            detections_path
        )

    if not embeddings_path.exists():
        raise FileNotFoundError(
            embeddings_path
        )

    detections = pd.read_csv(
        detections_path
    ).reset_index(drop=True)

    embeddings = np.load(
        embeddings_path
    )

    required_columns = {
        "frame",
        "camera_id",
        "detection_id",
        "matched_gt",
        "gt_person_id",
        "ground_x_m",
        "ground_y_m",
        "embedding_index",
    }

    missing_columns = (
        required_columns
        - set(detections.columns)
    )

    if missing_columns:
        raise ValueError(
            "В detections_with_features.csv "
            "отсутствуют поля: "
            f"{sorted(missing_columns)}"
        )

    if len(detections) != len(embeddings):
        raise ValueError(
            "Количество detections не совпадает "
            "с количеством embeddings: "
            f"{len(detections)} != "
            f"{len(embeddings)}"
        )

    detections["frame"] = (
        detections["frame"].astype(int)
    )

    detections["matched_gt"] = (
        detections["matched_gt"].astype(int)
    )

    detections["gt_person_id"] = (
        detections["gt_person_id"].astype(int)
    )

    detections = detections[
        detections["camera_id"].isin(
            [
                args.camera_a,
                args.camera_b,
            ]
        )
    ].reset_index(drop=True)

    frames = sorted(
        detections["frame"].unique()
    )

    if len(frames) < 2:
        raise ValueError(
            "Нужно не менее двух временных точек."
        )

    train_count = int(
        round(
            len(frames)
            * args.train_fraction
        )
    )

    train_count = max(
        1,
        min(
            train_count,
            len(frames) - 1,
        ),
    )

    train_frames = [
        int(value)
        for value in frames[:train_count]
    ]

    test_frames = [
        int(value)
        for value in frames[train_count:]
    ]

    geometry_thresholds = parse_float_list(
        args.geometry_thresholds
    )

    alphas = parse_float_list(
        args.alphas
    )

    acceptance_costs = parse_float_list(
        args.acceptance_costs
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_info = {
        "camera_a": args.camera_a,
        "camera_b": args.camera_b,
        "all_frames": [
            int(value)
            for value in frames
        ],
        "train_frames": train_frames,
        "test_frames": test_frames,
        "train_fraction": (
            args.train_fraction
        ),
    }

    with (
        args.output / "split.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            split_info,
            file,
            ensure_ascii=False,
            indent=2,
        )

    tuning_rows: list[dict] = []

    # Appearance-only.
    for acceptance_cost in acceptance_costs:
        metrics, _, _ = evaluate_frames(
            detections=detections,
            embeddings=embeddings,
            annotations_root=(
                args.annotations.resolve()
            ),
            frames=train_frames,
            camera_a=args.camera_a,
            camera_b=args.camera_b,
            method="appearance",
            geometry_threshold_m=None,
            alpha=None,
            acceptance_cost=(
                acceptance_cost
            ),
        )

        tuning_rows.append(metrics)

    # Geometry-only.
    for (
        geometry_threshold,
        acceptance_cost,
    ) in product(
        geometry_thresholds,
        acceptance_costs,
    ):
        metrics, _, _ = evaluate_frames(
            detections=detections,
            embeddings=embeddings,
            annotations_root=(
                args.annotations.resolve()
            ),
            frames=train_frames,
            camera_a=args.camera_a,
            camera_b=args.camera_b,
            method="geometry",
            geometry_threshold_m=(
                geometry_threshold
            ),
            alpha=None,
            acceptance_cost=(
                acceptance_cost
            ),
        )

        tuning_rows.append(metrics)

    # Fused appearance + geometry.
    for (
        geometry_threshold,
        alpha,
        acceptance_cost,
    ) in product(
        geometry_thresholds,
        alphas,
        acceptance_costs,
    ):
        metrics, _, _ = evaluate_frames(
            detections=detections,
            embeddings=embeddings,
            annotations_root=(
                args.annotations.resolve()
            ),
            frames=train_frames,
            camera_a=args.camera_a,
            camera_b=args.camera_b,
            method="fused",
            geometry_threshold_m=(
                geometry_threshold
            ),
            alpha=alpha,
            acceptance_cost=(
                acceptance_cost
            ),
        )

        tuning_rows.append(metrics)

    tuning_results = pd.DataFrame(
        tuning_rows
    )

    tuning_results.to_csv(
        args.output
        / "tuning_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected_rows: list[pd.Series] = []

    for method in [
        "appearance",
        "geometry",
        "fused",
    ]:
        selected_rows.append(
            choose_best(
                tuning_results[
                    tuning_results["method"]
                    == method
                ]
            )
        )

    selected_parameters = pd.DataFrame(
        selected_rows
    )

    selected_parameters.to_csv(
        args.output
        / "selected_parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_metrics_rows: list[dict] = []
    full_metrics_rows: list[dict] = []

    all_test_frame_rows: list[
        pd.DataFrame
    ] = []

    all_test_pair_rows: list[
        pd.DataFrame
    ] = []

    for selected in selected_rows:
        method = str(selected["method"])

        geometry_threshold = (
            None
            if pd.isna(
                selected[
                    "geometry_threshold_m"
                ]
            )
            else float(
                selected[
                    "geometry_threshold_m"
                ]
            )
        )

        alpha = (
            None
            if pd.isna(selected["alpha"])
            else float(selected["alpha"])
        )

        acceptance_cost = float(
            selected["acceptance_cost"]
        )

        (
            test_metrics,
            test_frame_rows,
            test_pair_rows,
        ) = evaluate_frames(
            detections=detections,
            embeddings=embeddings,
            annotations_root=(
                args.annotations.resolve()
            ),
            frames=test_frames,
            camera_a=args.camera_a,
            camera_b=args.camera_b,
            method=method,
            geometry_threshold_m=(
                geometry_threshold
            ),
            alpha=alpha,
            acceptance_cost=(
                acceptance_cost
            ),
            keep_pair_rows=True,
        )

        test_metrics["split"] = "test"
        test_metrics_rows.append(
            test_metrics
        )

        test_frame_rows["method"] = method
        test_frame_rows[
            "geometry_threshold_m"
        ] = geometry_threshold
        test_frame_rows["alpha"] = alpha
        test_frame_rows[
            "acceptance_cost"
        ] = acceptance_cost

        test_pair_rows["method"] = method
        test_pair_rows[
            "geometry_threshold_m"
        ] = geometry_threshold
        test_pair_rows["alpha"] = alpha
        test_pair_rows[
            "acceptance_cost"
        ] = acceptance_cost

        all_test_frame_rows.append(
            test_frame_rows
        )

        all_test_pair_rows.append(
            test_pair_rows
        )

        full_metrics, _, _ = evaluate_frames(
            detections=detections,
            embeddings=embeddings,
            annotations_root=(
                args.annotations.resolve()
            ),
            frames=[
                int(value)
                for value in frames
            ],
            camera_a=args.camera_a,
            camera_b=args.camera_b,
            method=method,
            geometry_threshold_m=(
                geometry_threshold
            ),
            alpha=alpha,
            acceptance_cost=(
                acceptance_cost
            ),
        )

        full_metrics["split"] = "full"
        full_metrics_rows.append(
            full_metrics
        )

    test_summary = pd.DataFrame(
        test_metrics_rows
    ).sort_values(
        "f1",
        ascending=False,
    )

    full_summary = pd.DataFrame(
        full_metrics_rows
    ).sort_values(
        "f1",
        ascending=False,
    )

    test_summary.to_csv(
        args.output
        / "test_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    full_summary.to_csv(
        args.output
        / "full_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(
        all_test_frame_rows,
        ignore_index=True,
    ).to_csv(
        args.output
        / "test_per_frame.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(
        all_test_pair_rows,
        ignore_index=True,
    ).to_csv(
        args.output
        / "test_assigned_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Разделение кадров:")
    print(
        f"  train: {len(train_frames)}"
    )
    print(
        f"  test:  {len(test_frames)}"
    )

    print()
    print(
        "Выбранные параметры "
        "(по train F1):"
    )
    print(
        selected_parameters[
            [
                "method",
                "geometry_threshold_m",
                "alpha",
                "acceptance_cost",
                "precision",
                "recall",
                "f1",
                "conditional_accuracy",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print()
    print("Результаты на test:")
    print(
        test_summary[
            [
                "method",
                "geometry_threshold_m",
                "alpha",
                "acceptance_cost",
                "pair_detection_coverage",
                "precision",
                "recall",
                "f1",
                "conditional_accuracy",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print()
    print(
        "Результаты:",
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
