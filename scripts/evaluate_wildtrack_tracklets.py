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
            "Оценка внутрикамерных tracklets WILDTRACK: "
            "чистота треков, фрагментации и приближённые ID-switches."
        )
    )

    parser.add_argument(
        "--observations",
        type=Path,
        required=True,
        help="Путь к observations.csv после ByteTrack.",
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(
            "data/raw/WILDTRACK/annotations_positions"
        ),
    )

    parser.add_argument(
        "--camera",
        required=True,
        help="Камера, например C1.",
    )

    parser.add_argument(
        "--thresholds",
        default="0.30,0.50",
        help="Пороги IoU для оценки.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
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


def normalize_frame_name(value: object) -> str:
    """
    Pandas может прочитать 00000005 как число 5.
    Возвращаем имя аннотации в формате 00000005.
    """
    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    try:
        number = int(text)
    except ValueError as error:
        raise ValueError(
            f"Некорректное имя кадра: {value!r}"
        ) from error

    return f"{number:08d}"


def load_gt_for_frame(
    annotation_root: Path,
    frame_name: str,
    camera_name: str,
) -> pd.DataFrame:
    annotation_path = (
        annotation_root / f"{frame_name}.json"
    )

    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Аннотация не найдена: {annotation_path.resolve()}"
        )

    with annotation_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        annotations = json.load(file)

    view_number = camera_to_view_number(
        camera_name
    )

    rows: list[dict] = []

    for person in annotations:
        views = {
            int(view["viewNum"]): view
            for view in person.get("views", [])
        }

        view = views.get(view_number)

        if view is None or not is_valid_view(view):
            continue

        rows.append(
            {
                "person_id": int(
                    person["personID"]
                ),
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

    inter_x1 = np.maximum(
        a[..., 0],
        b[..., 0],
    )
    inter_y1 = np.maximum(
        a[..., 1],
        b[..., 1],
    )
    inter_x2 = np.minimum(
        a[..., 2],
        b[..., 2],
    )
    inter_y2 = np.minimum(
        a[..., 3],
        b[..., 3],
    )

    inter_width = np.maximum(
        0.0,
        inter_x2 - inter_x1,
    )
    inter_height = np.maximum(
        0.0,
        inter_y2 - inter_y1,
    )

    intersection = (
        inter_width * inter_height
    )

    area_a = np.maximum(
        0.0,
        (
            a[..., 2] - a[..., 0]
        )
        * (
            a[..., 3] - a[..., 1]
        ),
    )

    area_b = np.maximum(
        0.0,
        (
            b[..., 2] - b[..., 0]
        )
        * (
            b[..., 3] - b[..., 1]
        ),
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def match_frame(
    frame_observations: pd.DataFrame,
    frame_gt: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, dict]:
    observations = (
        frame_observations
        .copy()
        .reset_index(drop=True)
    )

    observations[
        "matched_person_id"
    ] = -1
    observations["match_iou"] = 0.0

    gt_count = len(frame_gt)
    observation_count = len(observations)

    if observation_count == 0:
        return observations, {
            "gt_count": gt_count,
            "observation_count": 0,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": gt_count,
        }

    if gt_count == 0:
        return observations, {
            "gt_count": 0,
            "observation_count": (
                observation_count
            ),
            "true_positive": 0,
            "false_positive": (
                observation_count
            ),
            "false_negative": 0,
        }

    observation_boxes = observations[
        ["x1", "y1", "x2", "y2"]
    ].to_numpy(dtype=np.float64)

    gt_boxes = frame_gt[
        ["x1", "y1", "x2", "y2"]
    ].to_numpy(dtype=np.float64)

    iou_matrix = pairwise_iou(
        observation_boxes,
        gt_boxes,
    )

    observation_indices, gt_indices = (
        linear_sum_assignment(
            1.0 - iou_matrix
        )
    )

    true_positive = 0

    for observation_index, gt_index in zip(
        observation_indices,
        gt_indices,
    ):
        value = float(
            iou_matrix[
                observation_index,
                gt_index,
            ]
        )

        if value < threshold:
            continue

        observations.loc[
            observation_index,
            "matched_person_id",
        ] = int(
            frame_gt.iloc[
                gt_index
            ]["person_id"]
        )

        observations.loc[
            observation_index,
            "match_iou",
        ] = value

        true_positive += 1

    false_positive = (
        observation_count
        - true_positive
    )
    false_negative = (
        gt_count
        - true_positive
    )

    return observations, {
        "gt_count": gt_count,
        "observation_count": (
            observation_count
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
    }


def build_track_purity(
    matched_observations: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for track_id, group in (
        matched_observations.groupby(
            "track_id",
            sort=True,
        )
    ):
        matched = group[
            group["matched_person_id"] >= 0
        ]

        total_observations = len(group)
        matched_count = len(matched)

        if matched_count > 0:
            counts = (
                matched[
                    "matched_person_id"
                ]
                .astype(int)
                .value_counts()
            )

            dominant_person_id = int(
                counts.index[0]
            )
            dominant_count = int(
                counts.iloc[0]
            )

            purity = (
                dominant_count
                / matched_count
            )

            total_purity = (
                dominant_count
                / total_observations
            )

            unique_person_ids = int(
                counts.size
            )
        else:
            dominant_person_id = -1
            dominant_count = 0
            purity = 0.0
            total_purity = 0.0
            unique_person_ids = 0

        ordered = group.sort_values(
            "frame_index"
        )

        rows.append(
            {
                "track_id": int(track_id),
                "first_frame_index": int(
                    ordered[
                        "frame_index"
                    ].min()
                ),
                "last_frame_index": int(
                    ordered[
                        "frame_index"
                    ].max()
                ),
                "observations": int(
                    total_observations
                ),
                "matched_observations": int(
                    matched_count
                ),
                "matched_fraction": (
                    matched_count
                    / total_observations
                    if total_observations > 0
                    else 0.0
                ),
                "dominant_person_id": (
                    dominant_person_id
                ),
                "dominant_count": (
                    dominant_count
                ),
                "unique_person_ids": (
                    unique_person_ids
                ),
                "purity_on_matched": (
                    purity
                ),
                "purity_on_all": (
                    total_purity
                ),
                "mean_match_iou": float(
                    matched[
                        "match_iou"
                    ].mean()
                )
                if matched_count > 0
                else 0.0,
            }
        )

    return pd.DataFrame(rows)


def build_person_fragmentation(
    matched_observations: pd.DataFrame,
) -> pd.DataFrame:
    matched = matched_observations[
        matched_observations[
            "matched_person_id"
        ] >= 0
    ].copy()

    if matched.empty:
        return pd.DataFrame(
            columns=[
                "person_id",
                "observations",
                "unique_track_ids",
                "fragments",
                "approx_id_switches",
                "dominant_track_id",
                "dominant_track_fraction",
            ]
        )

    rows: list[dict] = []

    for person_id, group in matched.groupby(
        "matched_person_id",
        sort=True,
    ):
        ordered = group.sort_values(
            ["frame_index", "track_id"]
        )

        track_counts = (
            ordered["track_id"]
            .astype(int)
            .value_counts()
        )

        unique_track_ids = int(
            track_counts.size
        )

        dominant_track_id = int(
            track_counts.index[0]
        )

        dominant_count = int(
            track_counts.iloc[0]
        )

        track_sequence = (
            ordered["track_id"]
            .astype(int)
            .to_numpy()
        )

        approx_id_switches = int(
            np.sum(
                track_sequence[1:]
                != track_sequence[:-1]
            )
        )

        rows.append(
            {
                "person_id": int(
                    person_id
                ),
                "observations": int(
                    len(ordered)
                ),
                "unique_track_ids": (
                    unique_track_ids
                ),
                "fragments": max(
                    0,
                    unique_track_ids - 1,
                ),
                "approx_id_switches": (
                    approx_id_switches
                ),
                "dominant_track_id": (
                    dominant_track_id
                ),
                "dominant_track_fraction": (
                    dominant_count
                    / len(ordered)
                ),
                "first_frame_index": int(
                    ordered[
                        "frame_index"
                    ].min()
                ),
                "last_frame_index": int(
                    ordered[
                        "frame_index"
                    ].max()
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_threshold(
    threshold: float,
    frame_metrics: pd.DataFrame,
    track_purity: pd.DataFrame,
    person_fragmentation: pd.DataFrame,
) -> dict:
    gt_count = int(
        frame_metrics[
            "gt_count"
        ].sum()
    )

    observation_count = int(
        frame_metrics[
            "observation_count"
        ].sum()
    )

    true_positive = int(
        frame_metrics[
            "true_positive"
        ].sum()
    )

    false_positive = int(
        frame_metrics[
            "false_positive"
        ].sum()
    )

    false_negative = int(
        frame_metrics[
            "false_negative"
        ].sum()
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
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        ) > 0
        else 0.0
    )

    matched_tracks = track_purity[
        track_purity[
            "matched_observations"
        ] > 0
    ]

    return {
        "match_iou_threshold": (
            threshold
        ),
        "frames": int(
            len(frame_metrics)
        ),
        "gt_observations": (
            gt_count
        ),
        "tracker_observations": (
            observation_count
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
        "total_tracks": int(
            len(track_purity)
        ),
        "matched_tracks": int(
            len(matched_tracks)
        ),
        "tracks_purity_ge_0_80": int(
            (
                matched_tracks[
                    "purity_on_matched"
                ] >= 0.80
            ).sum()
        ),
        "tracks_purity_ge_0_90": int(
            (
                matched_tracks[
                    "purity_on_matched"
                ] >= 0.90
            ).sum()
        ),
        "mean_track_purity": float(
            matched_tracks[
                "purity_on_matched"
            ].mean()
        )
        if not matched_tracks.empty
        else 0.0,
        "median_track_purity": float(
            matched_tracks[
                "purity_on_matched"
            ].median()
        )
        if not matched_tracks.empty
        else 0.0,
        "gt_persons_matched": int(
            len(person_fragmentation)
        ),
        "total_fragments": int(
            person_fragmentation[
                "fragments"
            ].sum()
        )
        if not person_fragmentation.empty
        else 0,
        "mean_tracklets_per_person": float(
            person_fragmentation[
                "unique_track_ids"
            ].mean()
        )
        if not person_fragmentation.empty
        else 0.0,
        "median_tracklets_per_person": float(
            person_fragmentation[
                "unique_track_ids"
            ].median()
        )
        if not person_fragmentation.empty
        else 0.0,
        "max_tracklets_per_person": int(
            person_fragmentation[
                "unique_track_ids"
            ].max()
        )
        if not person_fragmentation.empty
        else 0,
        "approx_id_switches": int(
            person_fragmentation[
                "approx_id_switches"
            ].sum()
        )
        if not person_fragmentation.empty
        else 0,
        "mean_dominant_track_fraction": float(
            person_fragmentation[
                "dominant_track_fraction"
            ].mean()
        )
        if not person_fragmentation.empty
        else 0.0,
    }


def main() -> None:
    args = parse_args()

    observations_path = (
        args.observations.resolve()
    )

    annotation_root = (
        args.annotations.resolve()
    )

    if not observations_path.exists():
        raise FileNotFoundError(
            observations_path
        )

    observations = pd.read_csv(
        observations_path
    )

    required_columns = {
        "frame_index",
        "frame_name",
        "track_id",
        "x1",
        "y1",
        "x2",
        "y2",
    }

    missing_columns = (
        required_columns
        - set(observations.columns)
    )

    if missing_columns:
        raise ValueError(
            "В observations.csv отсутствуют поля: "
            f"{sorted(missing_columns)}"
        )

    observations[
        "frame_index"
    ] = observations[
        "frame_index"
    ].astype(int)

    observations[
        "track_id"
    ] = observations[
        "track_id"
    ].astype(int)

    observations[
        "normalized_frame_name"
    ] = observations[
        "frame_name"
    ].apply(
        normalize_frame_name
    )

    thresholds = [
        float(value.strip())
        for value in args.thresholds.split(",")
        if value.strip()
    ]

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_summary_rows: list[dict] = []

    unique_frames = (
        observations[
            [
                "frame_index",
                "normalized_frame_name",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "frame_index"
        )
    )

    for threshold in thresholds:
        matched_frames: list[
            pd.DataFrame
        ] = []

        frame_metric_rows: list[
            dict
        ] = []

        for frame_row in (
            unique_frames.itertuples(
                index=False
            )
        ):
            frame_index = int(
                frame_row.frame_index
            )

            frame_name = str(
                frame_row.normalized_frame_name
            )

            frame_observations = (
                observations[
                    observations[
                        "frame_index"
                    ] == frame_index
                ]
                .copy()
            )

            frame_gt = load_gt_for_frame(
                annotation_root=(
                    annotation_root
                ),
                frame_name=frame_name,
                camera_name=args.camera,
            )

            matched, metrics = (
                match_frame(
                    frame_observations=(
                        frame_observations
                    ),
                    frame_gt=frame_gt,
                    threshold=threshold,
                )
            )

            matched[
                "match_iou_threshold"
            ] = threshold

            matched_frames.append(
                matched
            )

            metrics.update(
                {
                    "frame_index": (
                        frame_index
                    ),
                    "frame_name": (
                        frame_name
                    ),
                    "match_iou_threshold": (
                        threshold
                    ),
                }
            )

            frame_metric_rows.append(
                metrics
            )

        matched_observations = (
            pd.concat(
                matched_frames,
                ignore_index=True,
            )
        )

        frame_metrics = pd.DataFrame(
            frame_metric_rows
        )

        track_purity = build_track_purity(
            matched_observations
        )

        person_fragmentation = (
            build_person_fragmentation(
                matched_observations
            )
        )

        threshold_name = (
            f"iou_{threshold:.2f}"
            .replace(".", "_")
        )

        matched_observations.to_csv(
            args.output
            / (
                f"observation_matches_"
                f"{threshold_name}.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )

        frame_metrics.to_csv(
            args.output
            / (
                f"frame_metrics_"
                f"{threshold_name}.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )

        track_purity.to_csv(
            args.output
            / (
                f"track_purity_"
                f"{threshold_name}.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )

        person_fragmentation.to_csv(
            args.output
            / (
                f"person_fragmentation_"
                f"{threshold_name}.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )

        summary = summarize_threshold(
            threshold=threshold,
            frame_metrics=frame_metrics,
            track_purity=track_purity,
            person_fragmentation=(
                person_fragmentation
            ),
        )

        all_summary_rows.append(
            summary
        )

    summary_table = pd.DataFrame(
        all_summary_rows
    )

    summary_table.to_csv(
        args.output / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Итог:")
    print(
        summary_table.round(4).to_string(
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
