from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Построение внутрикамерных tracklets для "
            "последовательности изображений WILDTRACK."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/WILDTRACK"),
    )

    parser.add_argument(
        "--camera",
        required=True,
        help="Камера WILDTRACK, например C1.",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=400,
        help="Количество первых изображений. 0 — все.",
    )

    parser.add_argument(
        "--model",
        default="yolo11n.pt",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
    )

    parser.add_argument(
        "--min-crop-width",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--min-crop-height",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--save-video",
        action="store_true",
    )

    parser.add_argument(
        "--video-fps",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def clip_box(
    box: list[float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box

    x1 = max(
        0,
        min(
            image_width - 1,
            int(round(x1)),
        ),
    )

    y1 = max(
        0,
        min(
            image_height - 1,
            int(round(y1)),
        ),
    )

    x2 = max(
        x1 + 1,
        min(
            image_width,
            int(round(x2)),
        ),
    )

    y2 = max(
        y1 + 1,
        min(
            image_height,
            int(round(y2)),
        ),
    )

    return x1, y1, x2, y2


def create_tracks_summary(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame(
            columns=[
                "camera_id",
                "track_id",
                "first_frame_index",
                "last_frame_index",
                "first_frame_name",
                "last_frame_name",
                "observations",
                "duration_frames",
                "mean_confidence",
                "max_confidence",
                "mean_bbox_width",
                "mean_bbox_height",
            ]
        )

    rows: list[dict] = []

    for track_id, group in observations.groupby(
        "track_id",
        sort=True,
    ):
        ordered = group.sort_values(
            "frame_index"
        )

        first_row = ordered.iloc[0]
        last_row = ordered.iloc[-1]

        rows.append(
            {
                "camera_id": str(
                    first_row["camera_id"]
                ),
                "track_id": int(track_id),
                "first_frame_index": int(
                    first_row["frame_index"]
                ),
                "last_frame_index": int(
                    last_row["frame_index"]
                ),
                "first_frame_name": str(
                    first_row["frame_name"]
                ),
                "last_frame_name": str(
                    last_row["frame_name"]
                ),
                "observations": int(
                    len(ordered)
                ),
                "duration_frames": int(
                    last_row["frame_index"]
                    - first_row["frame_index"]
                    + 1
                ),
                "mean_confidence": float(
                    ordered["confidence"].mean()
                ),
                "max_confidence": float(
                    ordered["confidence"].max()
                ),
                "mean_bbox_width": float(
                    ordered["bbox_width"].mean()
                ),
                "mean_bbox_height": float(
                    ordered["bbox_height"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    image_dir = (
        args.dataset_root
        / "Image_subsets"
        / args.camera
    )

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Папка камеры не найдена: "
            f"{image_dir.resolve()}"
        )

    image_paths = sorted(
        [
            *image_dir.glob("*.png"),
            *image_dir.glob("*.jpg"),
            *image_dir.glob("*.jpeg"),
        ]
    )

    if args.frames > 0:
        image_paths = image_paths[
            : args.frames
        ]

    if not image_paths:
        raise FileNotFoundError(
            f"Изображения не найдены: "
            f"{image_dir.resolve()}"
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    crops_root = (
        args.output
        / "crops"
        / args.camera
    )

    crops_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    device: str | int = (
        0
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Camera: {args.camera}")
    print(f"Images: {len(image_paths)}")
    print(f"Model: {args.model}")
    print(f"Tracker: {args.tracker}")
    print(f"Device: {device}")

    model = YOLO(args.model)

    video_writer = None
    video_path = (
        args.output
        / f"{args.camera}_annotated.mp4"
    )

    observation_rows: list[dict] = []

    for frame_index, image_path in enumerate(
        image_paths
    ):
        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise FileNotFoundError(
                f"Не удалось прочитать: "
                f"{image_path.resolve()}"
            )

        image_height, image_width = (
            image.shape[:2]
        )

        result = model.track(
            source=image,
            persist=True,
            classes=[0],
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            tracker=args.tracker,
            device=device,
            verbose=False,
        )[0]

        if args.save_video:
            annotated = result.plot()

            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                video_writer = cv2.VideoWriter(
                    str(video_path),
                    fourcc,
                    args.video_fps,
                    (
                        image_width,
                        image_height,
                    ),
                )

                if not video_writer.isOpened():
                    raise RuntimeError(
                        "Не удалось открыть "
                        "VideoWriter."
                    )

            video_writer.write(
                annotated
            )

        boxes = result.boxes

        if (
            boxes is None
            or boxes.id is None
            or len(boxes) == 0
        ):
            print(
                f"Processed: "
                f"{frame_index + 1}/"
                f"{len(image_paths)}",
                end="\r",
            )
            continue

        xyxy = (
            boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )

        confidences = (
            boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        track_ids = (
            boxes.id
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

        frame_name = image_path.stem

        for (
            box,
            confidence,
            track_id,
        ) in zip(
            xyxy,
            confidences,
            track_ids,
        ):
            x1, y1, x2, y2 = clip_box(
                box.tolist(),
                image_width=image_width,
                image_height=image_height,
            )

            bbox_width = x2 - x1
            bbox_height = y2 - y1

            crop_relative_path = ""

            if (
                bbox_width
                >= args.min_crop_width
                and bbox_height
                >= args.min_crop_height
            ):
                track_crop_dir = (
                    crops_root
                    / f"track_{track_id:05d}"
                )

                track_crop_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                crop_path = (
                    track_crop_dir
                    / f"{frame_name}.jpg"
                )

                crop = image[
                    y1:y2,
                    x1:x2,
                ]

                if crop.size > 0:
                    success = cv2.imwrite(
                        str(crop_path),
                        crop,
                    )

                    if success:
                        crop_relative_path = (
                            crop_path
                            .relative_to(
                                args.output
                            )
                            .as_posix()
                        )

            observation_rows.append(
                {
                    "camera_id": (
                        args.camera
                    ),
                    "frame_index": (
                        frame_index
                    ),
                    "frame_name": (
                        frame_name
                    ),
                    "track_id": int(
                        track_id
                    ),
                    "confidence": float(
                        confidence
                    ),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "bbox_width": (
                        bbox_width
                    ),
                    "bbox_height": (
                        bbox_height
                    ),
                    "foot_u": (
                        x1 + x2
                    ) / 2.0,
                    "foot_v": float(
                        y2
                    ),
                    "crop_path": (
                        crop_relative_path
                    ),
                }
            )

        print(
            f"Processed: "
            f"{frame_index + 1}/"
            f"{len(image_paths)}",
            end="\r",
        )

    print()

    if video_writer is not None:
        video_writer.release()

    observations = pd.DataFrame(
        observation_rows
    )

    tracks_summary = (
        create_tracks_summary(
            observations
        )
    )

    observations_path = (
        args.output
        / "observations.csv"
    )

    tracks_summary_path = (
        args.output
        / "tracks_summary.csv"
    )

    observations.to_csv(
        observations_path,
        index=False,
        encoding="utf-8-sig",
    )

    tracks_summary.to_csv(
        tracks_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    config = {
        "camera": args.camera,
        "images": len(image_paths),
        "model": args.model,
        "imgsz": args.imgsz,
        "confidence_threshold": (
            args.conf
        ),
        "nms_iou_threshold": (
            args.iou
        ),
        "tracker": args.tracker,
        "device": str(device),
        "save_video": (
            args.save_video
        ),
        "video_fps": (
            args.video_fps
        ),
    }

    with (
        args.output
        / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Observations: "
        f"{len(observations)}"
    )
    print(
        f"Tracks: "
        f"{len(tracks_summary)}"
    )

    if not tracks_summary.empty:
        print(
            "Tracks with >= 5 observations:",
            int(
                (
                    tracks_summary[
                        "observations"
                    ] >= 5
                ).sum()
            ),
        )

        print(
            "Median observations per track:",
            float(
                tracks_summary[
                    "observations"
                ].median()
            ),
        )

    print(
        "Results:",
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
