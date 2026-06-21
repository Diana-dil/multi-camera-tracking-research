from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DATASET_ROOT = Path("data/raw/WILDTRACK")
IMAGES_ROOT = DATASET_ROOT / "Image_subsets"
ANNOTATIONS_ROOT = DATASET_ROOT / "annotations_positions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Визуализация эталонных global ID WILDTRACK."
    )
    parser.add_argument(
        "--frame",
        default="00000000",
        help="Имя синхронного кадра без расширения.",
    )
    parser.add_argument(
        "--camera-a",
        default="C1",
        help="Первая камера.",
    )
    parser.add_argument(
        "--camera-b",
        default="C4",
        help="Вторая камера.",
    )
    parser.add_argument(
        "--output",
        default="results/wildtrack_gt_C1_C4_00000000.jpg",
        help="Путь сохранения результата.",
    )
    return parser.parse_args()


def get_view_number(camera_name: str, zero_based: bool) -> int:
    camera_index = int(camera_name.removeprefix("C"))

    if zero_based:
        return camera_index - 1

    return camera_index


def make_color(person_id: int) -> tuple[int, int, int]:
    """
    Детерминированный цвет для одного global ID.
    Один человек будет иметь одинаковый цвет на всех камерах.
    """
    rng = np.random.default_rng(person_id)
    values = rng.integers(70, 256, size=3)

    return int(values[0]), int(values[1]), int(values[2])


def resize_to_width(
    image: np.ndarray,
    target_width: int = 960,
) -> np.ndarray:
    height, width = image.shape[:2]

    scale = target_width / width
    target_height = int(height * scale)

    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )


def draw_boxes(
    image: np.ndarray,
    boxes: list[dict],
    camera_name: str,
    common_ids: set[int],
) -> np.ndarray:
    result = image.copy()

    for item in boxes:
        person_id = item["person_id"]
        x1, y1, x2, y2 = item["bbox"]

        color = make_color(person_id)

        thickness = 4 if person_id in common_ids else 2

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )

        label = f"GID {person_id}"

        if person_id in common_ids:
            label += " COMMON"

        text_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            2,
        )

        text_width, text_height = text_size

        text_y1 = max(0, y1 - text_height - baseline - 8)
        text_y2 = text_y1 + text_height + baseline + 8
        text_x2 = min(result.shape[1], x1 + text_width + 10)

        cv2.rectangle(
            result,
            (x1, text_y1),
            (text_x2, text_y2),
            color,
            -1,
        )

        cv2.putText(
            result,
            label,
            (x1 + 5, text_y2 - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    header = (
        f"{camera_name}: visible={len(boxes)}, "
        f"common={len(common_ids)}"
    )

    cv2.rectangle(
        result,
        (0, 0),
        (650, 55),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        result,
        header,
        (15, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return result


def main() -> None:
    args = parse_args()

    annotation_path = ANNOTATIONS_ROOT / f"{args.frame}.json"

    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Файл аннотации не найден: {annotation_path.resolve()}"
        )

    with annotation_path.open("r", encoding="utf-8") as file:
        annotations = json.load(file)

    if not isinstance(annotations, list):
        raise ValueError(
            "Ожидался список объектов в JSON-аннотации."
        )

    all_view_numbers = {
        int(view["viewNum"])
        for person in annotations
        for view in person.get("views", [])
        if "viewNum" in view
    }

    if not all_view_numbers:
        raise ValueError("В аннотации не найдены viewNum.")

    zero_based = min(all_view_numbers) == 0

    camera_names = [args.camera_a, args.camera_b]

    camera_view_numbers = {
        camera_name: get_view_number(camera_name, zero_based)
        for camera_name in camera_names
    }

    boxes_by_camera: dict[str, list[dict]] = {
        camera_name: []
        for camera_name in camera_names
    }

    for person in annotations:
        person_id = int(person["personID"])

        for view in person.get("views", []):
            view_number = int(view["viewNum"])

            for camera_name, expected_view_number in camera_view_numbers.items():
                if view_number != expected_view_number:
                    continue

                x1 = int(view["xmin"])
                y1 = int(view["ymin"])
                x2 = int(view["xmax"])
                y2 = int(view["ymax"])

                # В WILDTRACK отрицательные координаты означают,
                # что человек не виден в этой камере.
                if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                    continue

                boxes_by_camera[camera_name].append(
                    {
                        "person_id": person_id,
                        "bbox": (x1, y1, x2, y2),
                    }
                )

    ids_a = {
        item["person_id"]
        for item in boxes_by_camera[args.camera_a]
    }

    ids_b = {
        item["person_id"]
        for item in boxes_by_camera[args.camera_b]
    }

    common_ids = ids_a & ids_b

    rendered_images: list[np.ndarray] = []

    for camera_name in camera_names:
        image_path = (
            IMAGES_ROOT
            / camera_name
            / f"{args.frame}.png"
        )

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(
                f"Не удалось открыть изображение: {image_path.resolve()}"
            )

        rendered = draw_boxes(
            image=image,
            boxes=boxes_by_camera[camera_name],
            camera_name=camera_name,
            common_ids=common_ids,
        )

        rendered_images.append(
            resize_to_width(rendered, target_width=960)
        )

    combined = np.hstack(rendered_images)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), combined):
        raise RuntimeError(
            f"Не удалось сохранить изображение: {output_path.resolve()}"
        )

    print(f"Кадр: {args.frame}")
    print(
        f"{args.camera_a}: "
        f"{len(boxes_by_camera[args.camera_a])} видимых людей"
    )
    print(
        f"{args.camera_b}: "
        f"{len(boxes_by_camera[args.camera_b])} видимых людей"
    )
    print(
        f"Общих global ID: {len(common_ids)}"
    )
    print(
        "Общие ID:",
        sorted(common_ids),
    )
    print(
        f"Результат: {output_path.resolve()}"
    )


if __name__ == "__main__":
    main()