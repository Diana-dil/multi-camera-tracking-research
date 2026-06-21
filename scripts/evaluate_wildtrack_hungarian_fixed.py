from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_wildtrack_hungarian import evaluate_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Оценка фиксированных параметров венгерского "
            "межкамерного сопоставления без повторной настройки."
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

    parser.add_argument("--camera-a", required=True)
    parser.add_argument("--camera-b", required=True)

    parser.add_argument(
        "--geometry-threshold",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Вес appearance-cost.",
    )

    parser.add_argument(
        "--acceptance-cost",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--split",
        choices=["train", "test", "full"],
        default="test",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    features_dir = args.features_dir.resolve()
    detections_path = (
        features_dir / "detections_with_features.csv"
    )
    embeddings_path = (
        features_dir / "embeddings.npy"
    )

    if not detections_path.exists():
        raise FileNotFoundError(detections_path)

    if not embeddings_path.exists():
        raise FileNotFoundError(embeddings_path)

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

    missing = required_columns - set(
        detections.columns
    )

    if missing:
        raise ValueError(
            "В detections_with_features.csv "
            f"отсутствуют поля: {sorted(missing)}"
        )

    if len(detections) != len(embeddings):
        raise ValueError(
            "Количество detections не совпадает "
            "с количеством embeddings: "
            f"{len(detections)} != {len(embeddings)}"
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
            [args.camera_a, args.camera_b]
        )
    ].reset_index(drop=True)

    frames = sorted(
        int(value)
        for value in detections["frame"].unique()
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
        min(train_count, len(frames) - 1),
    )

    train_frames = frames[:train_count]
    test_frames = frames[train_count:]

    if args.split == "train":
        selected_frames = train_frames
    elif args.split == "test":
        selected_frames = test_frames
    else:
        selected_frames = frames

    metrics, per_frame, assigned_pairs = (
        evaluate_frames(
            detections=detections,
            embeddings=embeddings,
            annotations_root=(
                args.annotations.resolve()
            ),
            frames=selected_frames,
            camera_a=args.camera_a,
            camera_b=args.camera_b,
            method="fused",
            geometry_threshold_m=(
                args.geometry_threshold
            ),
            alpha=args.alpha,
            acceptance_cost=(
                args.acceptance_cost
            ),
            keep_pair_rows=True,
        )
    )

    metrics["split"] = args.split
    metrics["source"] = (
        "fixed_parameters_without_retuning"
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = pd.DataFrame([metrics])

    summary.to_csv(
        args.output / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_frame.to_csv(
        args.output / "per_frame.csv",
        index=False,
        encoding="utf-8-sig",
    )

    assigned_pairs.to_csv(
        args.output / "assigned_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    split_info = {
        "camera_a": args.camera_a,
        "camera_b": args.camera_b,
        "split": args.split,
        "train_fraction": args.train_fraction,
        "train_frames": train_frames,
        "test_frames": test_frames,
        "evaluated_frames": selected_frames,
        "fixed_parameters": {
            "geometry_threshold_m": (
                args.geometry_threshold
            ),
            "alpha": args.alpha,
            "acceptance_cost": (
                args.acceptance_cost
            ),
        },
    }

    with (
        args.output / "config.json"
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

    print()
    print("Фиксированные параметры:")
    print(
        f"  geometry threshold: "
        f"{args.geometry_threshold:.2f} м"
    )
    print(f"  alpha: {args.alpha:.2f}")
    print(
        f"  acceptance cost: "
        f"{args.acceptance_cost:.2f}"
    )
    print()
    print("Результат:")
    print(
        summary[
            [
                "split",
                "frames",
                "common_gt_ids",
                "detectable_gt_ids",
                "pair_detection_coverage",
                "predicted_pairs",
                "true_positive",
                "false_positive",
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
        "Файлы сохранены:",
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
