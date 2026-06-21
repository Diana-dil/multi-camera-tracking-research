from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Создание split-screen видео WILDTRACK "
            "с одинаковыми global_id в двух камерах."
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
        "--fps",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--panel-width",
        type=int,
        default=960,
    )

    parser.add_argument(
        "--show",
        choices=["cross-camera", "all"],
        default="cross-camera",
        help=(
            "cross-camera — показывать только ID, "
            "связанные между камерами."
        ),
    )

    parser.add_argument(
        "--min-global-observations",
        type=int,
        default=5,
        help=(
            "Минимальное суммарное число наблюдений "
            "global_id для отображения."
        ),
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="0 — обработать все общие кадры.",
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
        number = int(text)
    except ValueError:
        return Path(text).stem

    return f"{number:08d}"


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

    for extension in (
        ".png",
        ".jpg",
        ".jpeg",
    ):
        path = image_dir / (
            frame_name + extension
        )

        if path.exists():
            return path

    raise FileNotFoundError(
        f"Не найден кадр {frame_name} "
        f"для камеры {camera}."
    )


def color_for_id(
    global_id: int,
) -> tuple[int, int, int]:
    """
    Детерминированный яркий BGR-цвет.
    """
    rng = np.random.default_rng(
        global_id * 1009 + 17
    )

    color = rng.integers(
        low=70,
        high=256,
        size=3,
        dtype=np.uint8,
    )

    return tuple(
        int(value)
        for value in color
    )


def draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.60
    thickness = 2

    (
        text_width,
        text_height,
    ), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    x = max(0, x)
    y = max(
        text_height + baseline + 4,
        y,
    )

    top_left = (
        x,
        y - text_height - baseline - 6,
    )

    bottom_right = (
        x + text_width + 8,
        y + 2,
    )

    cv2.rectangle(
        image,
        top_left,
        bottom_right,
        color,
        thickness=-1,
    )

    brightness = (
        0.299 * color[2]
        + 0.587 * color[1]
        + 0.114 * color[0]
    )

    text_color = (
        (0, 0, 0)
        if brightness > 150
        else (255, 255, 255)
    )

    cv2.putText(
        image,
        text,
        (
            x + 4,
            y - baseline - 2,
        ),
        font,
        font_scale,
        text_color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def annotate_panel(
    image: np.ndarray,
    rows: pd.DataFrame,
    camera: str,
    frame_name: str,
) -> np.ndarray:
    annotated = image.copy()

    for row in rows.itertuples(
        index=False
    ):
        global_id = int(
            row.global_id
        )

        x1 = int(round(float(row.x1)))
        y1 = int(round(float(row.y1)))
        x2 = int(round(float(row.x2)))
        y2 = int(round(float(row.y2)))

        x1 = max(
            0,
            min(
                annotated.shape[1] - 1,
                x1,
            ),
        )

        y1 = max(
            0,
            min(
                annotated.shape[0] - 1,
                y1,
            ),
        )

        x2 = max(
            x1 + 1,
            min(
                annotated.shape[1],
                x2,
            ),
        )

        y2 = max(
            y1 + 1,
            min(
                annotated.shape[0],
                y2,
            ),
        )

        color = color_for_id(
            global_id
        )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            thickness=3,
        )

        label = f"G{global_id}"

        draw_label(
            annotated,
            label,
            x1,
            y1,
            color,
        )

    header = (
        f"{camera} | frame {frame_name} | "
        f"global objects: {len(rows)}"
    )

    cv2.rectangle(
        annotated,
        (0, 0),
        (
            annotated.shape[1],
            42,
        ),
        (20, 20, 20),
        thickness=-1,
    )

    cv2.putText(
        annotated,
        header,
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )

    return annotated


def resize_panel(
    image: np.ndarray,
    panel_width: int,
) -> np.ndarray:
    scale = (
        panel_width
        / image.shape[1]
    )

    panel_height = int(
        round(
            image.shape[0]
            * scale
        )
    )

    return cv2.resize(
        image,
        (
            panel_width,
            panel_height,
        ),
        interpolation=cv2.INTER_AREA,
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

    required_observation_columns = {
        "camera_id",
        "frame_index",
        "frame_name",
        "global_id",
        "x1",
        "y1",
        "x2",
        "y2",
    }

    missing = (
        required_observation_columns
        - set(observations.columns)
    )

    if missing:
        raise ValueError(
            "В global_observations.csv "
            "отсутствуют поля: "
            f"{sorted(missing)}"
        )

    observations = observations[
        observations["camera_id"].isin(
            [
                args.camera_a,
                args.camera_b,
            ]
        )
    ].copy()

    observations[
        "frame_index"
    ] = observations[
        "frame_index"
    ].astype(int)

    observations[
        "global_id"
    ] = observations[
        "global_id"
    ].astype(int)

    observations[
        "normalized_frame_name"
    ] = observations[
        "frame_name"
    ].apply(
        normalize_frame_name
    )

    global_observation_counts = (
        observations.groupby(
            "global_id"
        )
        .size()
    )

    allowed_global_ids = set(
        global_observation_counts[
            global_observation_counts
            >= args.min_global_observations
        ].index.astype(int)
    )

    if args.show == "cross-camera":
        cross_camera_ids = set(
            tracks.loc[
                tracks[
                    "is_cross_camera"
                ] == 1,
                "global_id",
            ].astype(int)
        )

        allowed_global_ids &= (
            cross_camera_ids
        )

    observations = observations[
        observations[
            "global_id"
        ].isin(
            allowed_global_ids
        )
    ].copy()

    all_observations = pd.read_csv(
        observations_path
    )

    all_observations = all_observations[
        all_observations[
            "camera_id"
        ].isin(
            [
                args.camera_a,
                args.camera_b,
            ]
        )
    ].copy()

    all_observations[
        "frame_index"
    ] = all_observations[
        "frame_index"
    ].astype(int)

    frames_a = set(
        all_observations.loc[
            all_observations[
                "camera_id"
            ] == args.camera_a,
            "frame_index",
        ]
    )

    frames_b = set(
        all_observations.loc[
            all_observations[
                "camera_id"
            ] == args.camera_b,
            "frame_index",
        ]
    )

    common_frames = sorted(
        frames_a & frames_b
    )

    if args.max_frames > 0:
        common_frames = common_frames[
            : args.max_frames
        ]

    if not common_frames:
        raise ValueError(
            "Нет общих кадров."
        )

    first_frame = common_frames[0]

    first_rows = all_observations[
        (
            all_observations[
                "camera_id"
            ] == args.camera_a
        )
        & (
            all_observations[
                "frame_index"
            ] == first_frame
        )
    ]

    if first_rows.empty:
        raise ValueError(
            "Не удалось определить первый кадр."
        )

    first_name = normalize_frame_name(
        first_rows.iloc[0][
            "frame_name"
        ]
    )

    first_image = cv2.imread(
        str(
            find_image(
                dataset_root,
                args.camera_a,
                first_name,
            )
        )
    )

    if first_image is None:
        raise RuntimeError(
            "Не удалось прочитать первый кадр."
        )

    first_panel = resize_panel(
        first_image,
        args.panel_width,
    )

    panel_height = (
        first_panel.shape[0]
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(args.output),
        fourcc,
        args.fps,
        (
            args.panel_width * 2,
            panel_height,
        ),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Не удалось открыть VideoWriter."
        )

    try:
        for number, frame_index in enumerate(
            common_frames,
            start=1,
        ):
            rows_all_a = all_observations[
                (
                    all_observations[
                        "camera_id"
                    ] == args.camera_a
                )
                & (
                    all_observations[
                        "frame_index"
                    ] == frame_index
                )
            ]

            rows_all_b = all_observations[
                (
                    all_observations[
                        "camera_id"
                    ] == args.camera_b
                )
                & (
                    all_observations[
                        "frame_index"
                    ] == frame_index
                )
            ]

            if (
                rows_all_a.empty
                or rows_all_b.empty
            ):
                continue

            frame_name_a = normalize_frame_name(
                rows_all_a.iloc[0][
                    "frame_name"
                ]
            )

            frame_name_b = normalize_frame_name(
                rows_all_b.iloc[0][
                    "frame_name"
                ]
            )

            image_a = cv2.imread(
                str(
                    find_image(
                        dataset_root,
                        args.camera_a,
                        frame_name_a,
                    )
                )
            )

            image_b = cv2.imread(
                str(
                    find_image(
                        dataset_root,
                        args.camera_b,
                        frame_name_b,
                    )
                )
            )

            if (
                image_a is None
                or image_b is None
            ):
                raise RuntimeError(
                    f"Не удалось прочитать "
                    f"кадр {frame_index}."
                )

            rows_a = observations[
                (
                    observations[
                        "camera_id"
                    ] == args.camera_a
                )
                & (
                    observations[
                        "frame_index"
                    ] == frame_index
                )
            ]

            rows_b = observations[
                (
                    observations[
                        "camera_id"
                    ] == args.camera_b
                )
                & (
                    observations[
                        "frame_index"
                    ] == frame_index
                )
            ]

            panel_a = annotate_panel(
                image=image_a,
                rows=rows_a,
                camera=args.camera_a,
                frame_name=frame_name_a,
            )

            panel_b = annotate_panel(
                image=image_b,
                rows=rows_b,
                camera=args.camera_b,
                frame_name=frame_name_b,
            )

            panel_a = resize_panel(
                panel_a,
                args.panel_width,
            )

            panel_b = resize_panel(
                panel_b,
                args.panel_width,
            )

            if (
                panel_a.shape[0]
                != panel_height
            ):
                panel_a = cv2.resize(
                    panel_a,
                    (
                        args.panel_width,
                        panel_height,
                    ),
                )

            if (
                panel_b.shape[0]
                != panel_height
            ):
                panel_b = cv2.resize(
                    panel_b,
                    (
                        args.panel_width,
                        panel_height,
                    ),
                )

            combined = np.hstack(
                [
                    panel_a,
                    panel_b,
                ]
            )

            writer.write(
                combined
            )

            print(
                f"Video: "
                f"{number}/{len(common_frames)}",
                end="\r",
            )

    finally:
        writer.release()

    print()
    print(
        f"Displayed global IDs: "
        f"{len(allowed_global_ids)}"
    )
    print(
        f"Frames: {len(common_frames)}"
    )
    print(
        f"Video: {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
