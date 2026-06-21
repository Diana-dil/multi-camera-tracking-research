from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd


DATASET_ROOT = Path("data/raw/WILDTRACK")
ANNOTATIONS_ROOT = DATASET_ROOT / "annotations_positions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Анализ пересечения полей обзора камер WILDTRACK."
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=50,
        help="Количество первых синхронных кадров.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/wildtrack_overlap_50.csv"),
        help="Путь к итоговой таблице.",
    )
    return parser.parse_args()


def is_visible(view: dict) -> bool:
    """Проверяет, виден ли человек в камере."""
    x1 = int(view["xmin"])
    y1 = int(view["ymin"])
    x2 = int(view["xmax"])
    y2 = int(view["ymax"])

    return x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1


def detect_view_numbering(annotation_files: list[Path]) -> bool:
    """
    Возвращает True, если viewNum начинается с нуля.
    """
    with annotation_files[0].open("r", encoding="utf-8") as file:
        annotations = json.load(file)

    view_numbers = {
        int(view["viewNum"])
        for person in annotations
        for view in person.get("views", [])
        if "viewNum" in view
    }

    if not view_numbers:
        raise RuntimeError("В аннотациях не обнаружены viewNum.")

    return min(view_numbers) == 0


def camera_name(view_number: int, zero_based: bool) -> str:
    if zero_based:
        return f"C{view_number + 1}"

    return f"C{view_number}"


def load_visible_ids(
    annotation_path: Path,
    zero_based: bool,
) -> dict[str, set[int]]:
    """
    Возвращает множества global ID, видимых в каждой камере.
    """
    with annotation_path.open("r", encoding="utf-8") as file:
        annotations = json.load(file)

    ids_by_camera = {
        f"C{index}": set()
        for index in range(1, 8)
    }

    for person in annotations:
        person_id = int(person["personID"])

        for view in person.get("views", []):
            if not is_visible(view):
                continue

            name = camera_name(
                int(view["viewNum"]),
                zero_based=zero_based,
            )

            if name in ids_by_camera:
                ids_by_camera[name].add(person_id)

    return ids_by_camera


def main() -> None:
    args = parse_args()

    annotation_files = sorted(
        ANNOTATIONS_ROOT.glob("*.json")
    )

    if not annotation_files:
        raise FileNotFoundError(
            f"JSON-аннотации не найдены: "
            f"{ANNOTATIONS_ROOT.resolve()}"
        )

    selected_files = annotation_files[: args.frames]

    zero_based = detect_view_numbering(selected_files)

    cameras = [f"C{index}" for index in range(1, 8)]
    camera_pairs = list(itertools.combinations(cameras, 2))

    frame_rows: list[dict] = []

    for annotation_path in selected_files:
        ids_by_camera = load_visible_ids(
            annotation_path,
            zero_based=zero_based,
        )

        for camera_a, camera_b in camera_pairs:
            ids_a = ids_by_camera[camera_a]
            ids_b = ids_by_camera[camera_b]

            common_ids = ids_a & ids_b
            union_ids = ids_a | ids_b

            minimum_visible = min(len(ids_a), len(ids_b))

            jaccard = (
                len(common_ids) / len(union_ids)
                if union_ids
                else 0.0
            )

            overlap_coefficient = (
                len(common_ids) / minimum_visible
                if minimum_visible > 0
                else 0.0
            )

            frame_rows.append(
                {
                    "frame": annotation_path.stem,
                    "camera_a": camera_a,
                    "camera_b": camera_b,
                    "visible_a": len(ids_a),
                    "visible_b": len(ids_b),
                    "common_count": len(common_ids),
                    "jaccard": jaccard,
                    "overlap_coefficient": overlap_coefficient,
                    "common_ids": ",".join(
                        str(value)
                        for value in sorted(common_ids)
                    ),
                }
            )

    per_frame_df = pd.DataFrame(frame_rows)

    summary_rows: list[dict] = []

    for (camera_a, camera_b), group in per_frame_df.groupby(
        ["camera_a", "camera_b"]
    ):
        all_common_ids: set[int] = set()

        for value in group["common_ids"]:
            if not value:
                continue

            all_common_ids.update(
                int(item)
                for item in value.split(",")
                if item
            )

        summary_rows.append(
            {
                "camera_a": camera_a,
                "camera_b": camera_b,
                "frames": len(group),
                "mean_visible_a": group["visible_a"].mean(),
                "mean_visible_b": group["visible_b"].mean(),
                "mean_common": group["common_count"].mean(),
                "median_common": group["common_count"].median(),
                "max_common": group["common_count"].max(),
                "frames_with_common": int(
                    (group["common_count"] > 0).sum()
                ),
                "mean_jaccard": group["jaccard"].mean(),
                "mean_overlap_coefficient": (
                    group["overlap_coefficient"].mean()
                ),
                "unique_common_ids": len(all_common_ids),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values(
        by=[
            "mean_common",
            "mean_overlap_coefficient",
            "unique_common_ids",
        ],
        ascending=False,
    ).reset_index(drop=True)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig",
    )

    per_frame_output = args.output.with_name(
        f"{args.output.stem}_per_frame.csv"
    )

    per_frame_df.to_csv(
        per_frame_output,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Проанализировано кадров: {len(selected_files)}")
    print(f"Проанализировано пар камер: {len(camera_pairs)}")
    print()
    print("Пять пар с наибольшим средним пересечением:")
    print(
        summary_df[
            [
                "camera_a",
                "camera_b",
                "mean_common",
                "max_common",
                "mean_overlap_coefficient",
                "unique_common_ids",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )
    print()
    print(f"Итоговая таблица: {args.output.resolve()}")
    print(f"Покадровая таблица: {per_frame_output.resolve()}")


if __name__ == "__main__":
    main()