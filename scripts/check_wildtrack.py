from __future__ import annotations

import json
from pathlib import Path


DATASET_ROOT = Path("data/raw/WILDTRACK")
IMAGES_ROOT = DATASET_ROOT / "Image_subsets"
ANNOTATIONS_ROOT = DATASET_ROOT / "annotations_positions"
CALIBRATIONS_ROOT = DATASET_ROOT / "calibrations"


def main() -> None:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Папка WILDTRACK не найдена: {DATASET_ROOT.resolve()}"
        )

    print(f"WILDTRACK root: {DATASET_ROOT.resolve()}")
    print()

    camera_dirs = sorted(
        path for path in IMAGES_ROOT.iterdir()
        if path.is_dir()
    )

    if not camera_dirs:
        raise RuntimeError("В Image_subsets не найдены папки камер.")

    print("Камеры:")

    camera_files: dict[str, list[Path]] = {}

    for camera_dir in camera_dirs:
        files = sorted(
            path
            for path in camera_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

        camera_files[camera_dir.name] = files

        first_name = files[0].name if files else "нет файлов"
        last_name = files[-1].name if files else "нет файлов"

        print(
            f"  {camera_dir.name}: "
            f"{len(files)} изображений, "
            f"первое={first_name}, последнее={last_name}"
        )

    print()

    annotation_files = sorted(ANNOTATIONS_ROOT.glob("*.json"))

    print(f"JSON-аннотаций: {len(annotation_files)}")

    if annotation_files:
        first_annotation = annotation_files[0]

        print(f"Первая аннотация: {first_annotation.name}")

        with first_annotation.open("r", encoding="utf-8") as file:
            annotation_data = json.load(file)

        print(f"Тип содержимого JSON: {type(annotation_data).__name__}")

        if isinstance(annotation_data, list):
            print(f"Число объектов в первом JSON: {len(annotation_data)}")

            if annotation_data:
                print(
                    "Поля первого объекта:",
                    sorted(annotation_data[0].keys())
                )

        elif isinstance(annotation_data, dict):
            print(
                "Поля верхнего уровня:",
                sorted(annotation_data.keys())
            )

    print()

    calibration_files = sorted(
        path
        for path in CALIBRATIONS_ROOT.rglob("*")
        if path.is_file()
    )

    print(f"Файлов калибровки: {len(calibration_files)}")

    for path in calibration_files[:15]:
        print(f"  {path.relative_to(DATASET_ROOT)}")

    print()

    filename_sets = [
        {path.name for path in files}
        for files in camera_files.values()
        if files
    ]

    if filename_sets:
        common_names = sorted(set.intersection(*filename_sets))
        print(
            "Синхронных имён файлов, присутствующих во всех камерах:",
            len(common_names),
        )

        if common_names:
            print("Первый синхронный кадр:", common_names[0])


if __name__ == "__main__":
    main()