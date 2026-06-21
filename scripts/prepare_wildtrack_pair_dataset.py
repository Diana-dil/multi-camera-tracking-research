from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd


DATASET_ROOT = Path("data/raw/WILDTRACK")
IMAGES_ROOT = DATASET_ROOT / "Image_subsets"
ANNOTATIONS_ROOT = DATASET_ROOT / "annotations_positions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Подготовка межкамерного набора изображений WILDTRACK."
    )
    parser.add_argument("--camera-a", default="C1")
    parser.add_argument("--camera-b", default="C6")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/wildtrack_C1_C6_50"),
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Дополнительное поле вокруг bounding box.",
    )
    return parser.parse_args()


def is_valid_view(view: dict) -> bool:
    x1 = int(view["xmin"])
    y1 = int(view["ymin"])
    x2 = int(view["xmax"])
    y2 = int(view["ymax"])

    return x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1


def detect_zero_based(annotation_path: Path) -> bool:
    with annotation_path.open("r", encoding="utf-8") as file:
        annotations = json.load(file)

    view_numbers = {
        int(view["viewNum"])
        for person in annotations
        for view in person.get("views", [])
        if "viewNum" in view
    }

    if not view_numbers:
        raise RuntimeError("В аннотации отсутствуют viewNum.")

    return min(view_numbers) == 0


def camera_to_view(camera: str, zero_based: bool) -> int:
    camera_number = int(camera.removeprefix("C"))
    return camera_number - 1 if zero_based else camera_number


def expand_and_clip_bbox(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    margin: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox

    width = x2 - x1
    height = y2 - y1

    margin_x = int(width * margin)
    margin_y = int(height * margin)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(image_width, x2 + margin_x)
    y2 = min(image_height, y2 + margin_y)

    return x1, y1, x2, y2


def main() -> None:
    args = parse_args()

    annotation_files = sorted(
        ANNOTATIONS_ROOT.glob("*.json")
    )[: args.frames]

    if not annotation_files:
        raise FileNotFoundError(
            f"Аннотации не найдены: {ANNOTATIONS_ROOT.resolve()}"
        )

    zero_based = detect_zero_based(annotation_files[0])

    cameras = [args.camera_a, args.camera_b]

    view_numbers = {
        camera: camera_to_view(camera, zero_based)
        for camera in cameras
    }

    args.output.mkdir(parents=True, exist_ok=True)

    for camera in cameras:
        (args.output / "crops" / camera).mkdir(
            parents=True,
            exist_ok=True,
        )

    metadata_rows: list[dict] = []

    for annotation_path in annotation_files:
        frame_name = annotation_path.stem

        with annotation_path.open("r", encoding="utf-8") as file:
            annotations = json.load(file)

        images: dict[str, object] = {}

        for camera in cameras:
            image_path = (
                IMAGES_ROOT
                / camera
                / f"{frame_name}.png"
            )

            image = cv2.imread(str(image_path))

            if image is None:
                raise FileNotFoundError(
                    f"Не удалось прочитать {image_path.resolve()}"
                )

            images[camera] = image

        for person in annotations:
            person_id = int(person["personID"])

            views_by_number = {
                int(view["viewNum"]): view
                for view in person.get("views", [])
            }

            for camera in cameras:
                view = views_by_number.get(view_numbers[camera])

                if view is None or not is_valid_view(view):
                    continue

                image = images[camera]
                image_height, image_width = image.shape[:2]

                original_bbox = (
                    int(view["xmin"]),
                    int(view["ymin"]),
                    int(view["xmax"]),
                    int(view["ymax"]),
                )

                x1, y1, x2, y2 = expand_and_clip_bbox(
                    original_bbox,
                    image_width=image_width,
                    image_height=image_height,
                    margin=args.margin,
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = image[y1:y2, x1:x2]

                crop_relative_path = (
                    Path("crops")
                    / camera
                    / f"{frame_name}_pid_{person_id}.jpg"
                )

                crop_path = args.output / crop_relative_path

                if not cv2.imwrite(str(crop_path), crop):
                    raise RuntimeError(
                        f"Не удалось сохранить {crop_path.resolve()}"
                    )

                metadata_rows.append(
                    {
                        "frame": frame_name,
                        "camera_id": camera,
                        "person_id": person_id,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "crop_path": crop_relative_path.as_posix(),
                    }
                )

    metadata = pd.DataFrame(metadata_rows)

    if metadata.empty:
        raise RuntimeError("Не было извлечено ни одного изображения.")

    metadata_path = args.output / "metadata.csv"
    metadata.to_csv(
        metadata_path,
        index=False,
        encoding="utf-8-sig",
    )

    positive_rows: list[dict] = []

    for (frame, person_id), group in metadata.groupby(
        ["frame", "person_id"]
    ):
        rows_by_camera = {
            row["camera_id"]: row
            for _, row in group.iterrows()
        }

        if (
            args.camera_a not in rows_by_camera
            or args.camera_b not in rows_by_camera
        ):
            continue

        row_a = rows_by_camera[args.camera_a]
        row_b = rows_by_camera[args.camera_b]

        positive_rows.append(
            {
                "frame": frame,
                "person_id": person_id,
                "camera_a": args.camera_a,
                "camera_b": args.camera_b,
                "crop_a": row_a["crop_path"],
                "crop_b": row_b["crop_path"],
                "label": 1,
            }
        )

    positive_pairs = pd.DataFrame(positive_rows)

    positive_path = args.output / "positive_pairs.csv"
    positive_pairs.to_csv(
        positive_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Камеры: {args.camera_a} и {args.camera_b}")
    print(f"Обработано кадров: {len(annotation_files)}")
    print(f"Всего изображений людей: {len(metadata)}")
    print(
        "Уникальных global ID:",
        metadata["person_id"].nunique(),
    )
    print(
        "Положительных межкамерных пар:",
        len(positive_pairs),
    )
    print(
        "Уникальных общих global ID:",
        positive_pairs["person_id"].nunique(),
    )
    print(f"Metadata: {metadata_path.resolve()}")
    print(f"Positive pairs: {positive_path.resolve()}")


if __name__ == "__main__":
    main()