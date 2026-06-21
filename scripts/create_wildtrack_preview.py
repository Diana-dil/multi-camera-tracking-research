from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


DATASET_ROOT = Path("data/raw/WILDTRACK")
IMAGES_ROOT = DATASET_ROOT / "Image_subsets"
OUTPUT_PATH = Path("results/wildtrack_preview.jpg")


def resize_with_label(
    image: np.ndarray,
    label: str,
    width: int = 640,
) -> np.ndarray:
    height = int(image.shape[0] * width / image.shape[1])

    resized = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )

    cv2.rectangle(
        resized,
        (0, 0),
        (150, 45),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.putText(
        resized,
        label,
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return resized


def main() -> None:
    camera_dirs = sorted(
        path
        for path in IMAGES_ROOT.iterdir()
        if path.is_dir()
    )

    if len(camera_dirs) != 7:
        print(
            f"Предупреждение: найдено камер: {len(camera_dirs)}, "
            "ожидалось 7."
        )

    filename_sets = []

    for camera_dir in camera_dirs:
        names = {
            path.name
            for path in camera_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        }
        filename_sets.append(names)

    common_names = sorted(set.intersection(*filename_sets))

    if not common_names:
        raise RuntimeError(
            "Не найдено ни одного имени кадра, общего для всех камер."
        )

    frame_name = common_names[0]

    images = []

    for camera_dir in camera_dirs:
        image_path = camera_dir / frame_name
        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"Не удалось прочитать изображение: {image_path}"
            )

        images.append(
            resize_with_label(
                image,
                label=f"{camera_dir.name}: {frame_name}",
            )
        )

    tile_height, tile_width = images[0].shape[:2]

    columns = 3
    rows = math.ceil(len(images) / columns)

    canvas = np.zeros(
        (rows * tile_height, columns * tile_width, 3),
        dtype=np.uint8,
    )

    for index, image in enumerate(images):
        row = index // columns
        column = index % columns

        y1 = row * tile_height
        y2 = y1 + tile_height
        x1 = column * tile_width
        x2 = x1 + tile_width

        canvas[y1:y2, x1:x2] = image

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(OUTPUT_PATH), canvas):
        raise RuntimeError(
            f"Не удалось сохранить изображение: {OUTPUT_PATH}"
        )

    print(f"Синхронный кадр: {frame_name}")
    print(f"Результат сохранён: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()