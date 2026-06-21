from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Создание галереи для визуальной проверки "
            "межкамерных global_id WILDTRACK."
        )
    )

    parser.add_argument(
        "--global-observations",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--global-tracks",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/WILDTRACK"),
    )

    parser.add_argument(
        "--camera-a",
        default="C1",
    )

    parser.add_argument(
        "--camera-b",
        default="C3",
    )

    parser.add_argument(
        "--max-ids",
        type=int,
        default=30,
        help=(
            "Сколько наиболее подтверждённых "
            "межкамерных ID показать. 0 — все."
        ),
    )

    parser.add_argument(
        "--rows-per-page",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--crop-width",
        type=int,
        default=220,
    )

    parser.add_argument(
        "--crop-height",
        type=int,
        default=320,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def normalize_frame_name(value: object) -> str:
    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    try:
        return f"{int(text):08d}"
    except ValueError:
        return Path(text).stem


def find_image(
    dataset_root: Path,
    camera: str,
    frame_name: str,
) -> Path:
    image_dir = (
        dataset_root
        / "Image_subsets"
        / camera
    )

    for suffix in (".png", ".jpg", ".jpeg"):
        path = image_dir / f"{frame_name}{suffix}"

        if path.exists():
            return path

    raise FileNotFoundError(
        f"Не найден кадр {frame_name} камеры {camera}."
    )


def crop_person(
    dataset_root: Path,
    row: pd.Series,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    camera = str(row["camera_id"])
    frame_name = normalize_frame_name(
        row["frame_name"]
    )

    image_path = find_image(
        dataset_root,
        camera,
        frame_name,
    )

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"Не удалось прочитать {image_path}."
        )

    height, width = image.shape[:2]

    x1 = max(
        0,
        min(
            width - 1,
            int(round(float(row["x1"]))),
        ),
    )

    y1 = max(
        0,
        min(
            height - 1,
            int(round(float(row["y1"]))),
        ),
    )

    x2 = max(
        x1 + 1,
        min(
            width,
            int(round(float(row["x2"]))),
        ),
    )

    y2 = max(
        y1 + 1,
        min(
            height,
            int(round(float(row["y2"]))),
        ),
    )

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        crop = np.zeros(
            (target_height, target_width, 3),
            dtype=np.uint8,
        )

    scale = min(
        target_width / crop.shape[1],
        target_height / crop.shape[0],
    )

    resized_width = max(
        1,
        int(round(crop.shape[1] * scale)),
    )

    resized_height = max(
        1,
        int(round(crop.shape[0] * scale)),
    )

    resized = cv2.resize(
        crop,
        (
            resized_width,
            resized_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full(
        (
            target_height,
            target_width,
            3,
        ),
        235,
        dtype=np.uint8,
    )

    offset_x = (
        target_width - resized_width
    ) // 2

    offset_y = (
        target_height - resized_height
    ) // 2

    canvas[
        offset_y:offset_y + resized_height,
        offset_x:offset_x + resized_width,
    ] = resized

    return canvas


def select_pair(
    group_a: pd.DataFrame,
    group_b: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, bool]:
    frames_a = set(
        group_a["frame_index"].astype(int)
    )

    frames_b = set(
        group_b["frame_index"].astype(int)
    )

    common_frames = sorted(
        frames_a & frames_b
    )

    candidates: list[
        tuple[float, float, pd.Series, pd.Series]
    ] = []

    for frame_index in common_frames:
        rows_a = group_a[
            group_a["frame_index"]
            == frame_index
        ]

        rows_b = group_b[
            group_b["frame_index"]
            == frame_index
        ]

        for _, row_a in rows_a.iterrows():
            for _, row_b in rows_b.iterrows():
                confidence_score = (
                    float(row_a.get("confidence", 0.0))
                    + float(row_b.get("confidence", 0.0))
                )

                area_a = (
                    float(row_a["x2"])
                    - float(row_a["x1"])
                ) * (
                    float(row_a["y2"])
                    - float(row_a["y1"])
                )

                area_b = (
                    float(row_b["x2"])
                    - float(row_b["x1"])
                ) * (
                    float(row_b["y2"])
                    - float(row_b["y1"])
                )

                area_score = min(area_a, area_b)

                candidates.append(
                    (
                        confidence_score,
                        area_score,
                        row_a,
                        row_b,
                    )
                )

    if candidates:
        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
            ),
            reverse=True,
        )

        _, _, row_a, row_b = candidates[0]

        return row_a, row_b, True

    def best_individual(
        group: pd.DataFrame,
    ) -> pd.Series:
        ranked = group.copy()

        ranked["_area"] = (
            ranked["x2"] - ranked["x1"]
        ) * (
            ranked["y2"] - ranked["y1"]
        )

        ranked["_score"] = (
            ranked.get(
                "confidence",
                pd.Series(
                    0.0,
                    index=ranked.index,
                ),
            )
            + ranked["_area"]
            / max(
                float(ranked["_area"].max()),
                1.0,
            )
        )

        return ranked.sort_values(
            "_score",
            ascending=False,
        ).iloc[0]

    return (
        best_individual(group_a),
        best_individual(group_b),
        False,
    )


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.62,
    thickness: int = 2,
    color: tuple[int, int, int] = (20, 20, 20),
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()

    observations_path = (
        args.global_observations.resolve()
    )

    tracks_path = (
        args.global_tracks.resolve()
    )

    dataset_root = (
        args.dataset_root.resolve()
    )

    if not observations_path.exists():
        raise FileNotFoundError(
            observations_path
        )

    if not tracks_path.exists():
        raise FileNotFoundError(
            tracks_path
        )

    observations = pd.read_csv(
        observations_path
    )

    tracks = pd.read_csv(
        tracks_path
    )

    required_columns = {
        "global_id",
        "camera_id",
        "frame_index",
        "frame_name",
        "x1",
        "y1",
        "x2",
        "y2",
    }

    missing = (
        required_columns
        - set(observations.columns)
    )

    if missing:
        raise ValueError(
            "В global_observations.csv "
            f"отсутствуют поля: {sorted(missing)}"
        )

    observations["global_id"] = (
        observations["global_id"].astype(int)
    )

    observations["frame_index"] = (
        observations["frame_index"].astype(int)
    )

    cross_camera_ids = set(
        tracks.loc[
            tracks["is_cross_camera"] == 1,
            "global_id",
        ].astype(int)
    )

    observations = observations[
        observations["global_id"].isin(
            cross_camera_ids
        )
    ].copy()

    support_rows: list[dict] = []

    for global_id, group in observations.groupby(
        "global_id"
    ):
        group_a = group[
            group["camera_id"]
            == args.camera_a
        ]

        group_b = group[
            group["camera_id"]
            == args.camera_b
        ]

        if group_a.empty or group_b.empty:
            continue

        common_frames = (
            set(
                group_a[
                    "frame_index"
                ].astype(int)
            )
            & set(
                group_b[
                    "frame_index"
                ].astype(int)
            )
        )

        support_rows.append(
            {
                "global_id": int(global_id),
                "camera_a_observations": int(
                    len(group_a)
                ),
                "camera_b_observations": int(
                    len(group_b)
                ),
                "common_frames": int(
                    len(common_frames)
                ),
                "support_score": int(
                    len(common_frames)
                ),
            }
        )

    support = pd.DataFrame(
        support_rows
    ).sort_values(
        [
            "support_score",
            "camera_a_observations",
            "camera_b_observations",
        ],
        ascending=False,
    )

    if args.max_ids > 0:
        support = support.head(
            args.max_ids
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    review_rows: list[dict] = []
    rendered_rows: list[np.ndarray] = []

    row_height = (
        args.crop_height + 90
    )

    row_width = (
        160
        + args.crop_width * 2
        + 60
    )

    for item in support.itertuples(
        index=False
    ):
        global_id = int(
            item.global_id
        )

        group = observations[
            observations["global_id"]
            == global_id
        ]

        group_a = group[
            group["camera_id"]
            == args.camera_a
        ]

        group_b = group[
            group["camera_id"]
            == args.camera_b
        ]

        row_a, row_b, synchronized = (
            select_pair(
                group_a,
                group_b,
            )
        )

        crop_a = crop_person(
            dataset_root,
            row_a,
            args.crop_width,
            args.crop_height,
        )

        crop_b = crop_person(
            dataset_root,
            row_b,
            args.crop_width,
            args.crop_height,
        )

        canvas = np.full(
            (
                row_height,
                row_width,
                3,
            ),
            250,
            dtype=np.uint8,
        )

        put_text(
            canvas,
            f"G{global_id}",
            (18, 42),
            scale=0.95,
            thickness=2,
        )

        put_text(
            canvas,
            (
                f"sync: {'yes' if synchronized else 'no'}"
            ),
            (18, 72),
            scale=0.50,
            thickness=1,
        )

        x_a = 160
        x_b = (
            160 + args.crop_width + 40
        )
        y_crop = 55

        canvas[
            y_crop:y_crop + args.crop_height,
            x_a:x_a + args.crop_width,
        ] = crop_a

        canvas[
            y_crop:y_crop + args.crop_height,
            x_b:x_b + args.crop_width,
        ] = crop_b

        put_text(
            canvas,
            (
                f"{args.camera_a} | frame "
                f"{normalize_frame_name(row_a['frame_name'])}"
            ),
            (x_a, 32),
            scale=0.55,
            thickness=1,
        )

        put_text(
            canvas,
            (
                f"{args.camera_b} | frame "
                f"{normalize_frame_name(row_b['frame_name'])}"
            ),
            (x_b, 32),
            scale=0.55,
            thickness=1,
        )

        frame_index_a = int(
            row_a["frame_index"]
        )

        frame_index_b = int(
            row_b["frame_index"]
        )

        if synchronized:
            ground_distance = math.hypot(
                float(
                    row_a["ground_x_m"]
                )
                - float(
                    row_b["ground_x_m"]
                ),
                float(
                    row_a["ground_y_m"]
                )
                - float(
                    row_b["ground_y_m"]
                ),
            )
        else:
            ground_distance = float("nan")

        put_text(
            canvas,
            (
                f"common frames: {item.common_frames}"
            ),
            (18, row_height - 18),
            scale=0.52,
            thickness=1,
        )

        rendered_rows.append(
            canvas
        )

        review_rows.append(
            {
                "global_id": global_id,
                "camera_a": args.camera_a,
                "camera_b": args.camera_b,
                "camera_a_frame_index": (
                    frame_index_a
                ),
                "camera_b_frame_index": (
                    frame_index_b
                ),
                "synchronized_pair": int(
                    synchronized
                ),
                "common_frames": int(
                    item.common_frames
                ),
                "camera_a_observations": int(
                    item.camera_a_observations
                ),
                "camera_b_observations": int(
                    item.camera_b_observations
                ),
                "ground_distance_m": (
                    ground_distance
                ),
                "manual_verdict": "",
                "manual_comment": "",
            }
        )

    pages = math.ceil(
        len(rendered_rows)
        / args.rows_per_page
    )

    for page_index in range(pages):
        start = (
            page_index
            * args.rows_per_page
        )

        end = min(
            start + args.rows_per_page,
            len(rendered_rows),
        )

        page_rows = rendered_rows[
            start:end
        ]

        page = np.full(
            (
                row_height
                * len(page_rows),
                row_width,
                3,
            ),
            245,
            dtype=np.uint8,
        )

        for row_index, rendered in enumerate(
            page_rows
        ):
            y = row_index * row_height

            page[
                y:y + row_height,
                :,
            ] = rendered

            if row_index > 0:
                cv2.line(
                    page,
                    (0, y),
                    (row_width, y),
                    (190, 190, 190),
                    2,
                )

        page_path = (
            args.output
            / (
                f"review_page_"
                f"{page_index + 1:02d}.jpg"
            )
        )

        cv2.imwrite(
            str(page_path),
            page,
        )

    review_table = pd.DataFrame(
        review_rows
    )

    review_table.to_csv(
        args.output / "manual_review.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"Global IDs selected: "
        f"{len(review_table)}"
    )

    print(
        f"Pages created: {pages}"
    )

    print(
        "Results:",
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
