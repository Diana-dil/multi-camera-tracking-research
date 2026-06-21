from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


DATASET_ROOT = Path("data/raw/WILDTRACK")
ANNOTATIONS_ROOT = DATASET_ROOT / "annotations_positions"
CALIBRATIONS_ROOT = DATASET_ROOT / "calibrations"


CAMERA_CALIBRATION_NAMES = {
    "C1": "CVLab1",
    "C2": "CVLab2",
    "C3": "CVLab3",
    "C4": "CVLab4",
    "C5": "IDIAP1",
    "C6": "IDIAP2",
    "C7": "IDIAP3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Проверка геометрической калибровки WILDTRACK "
            "по эталонным positionID."
        )
    )

    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["C1", "C6"],
        help="Камеры для проверки.",
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
        default=Path(
            "results/wildtrack_geometry_validation_C1_C6"
        ),
    )

    return parser.parse_args()


def read_xml_values(
    root: ET.Element,
    field_name: str,
) -> np.ndarray:
    node = root.find(field_name)

    if node is None:
        raise KeyError(
            f"Поле {field_name!r} не найдено в XML."
        )

    data_node = node.find("data")

    if data_node is not None:
        text = data_node.text
    else:
        text = node.text

    if not text:
        raise ValueError(
            f"Поле {field_name!r} не содержит значений."
        )

    values = np.array(
        [float(value) for value in text.split()],
        dtype=np.float64,
    )

    return values


def load_calibration(
    camera_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if camera_name not in CAMERA_CALIBRATION_NAMES:
        raise ValueError(
            f"Неизвестная камера: {camera_name}"
        )

    calibration_name = CAMERA_CALIBRATION_NAMES[
        camera_name
    ]

    intrinsic_path = (
        CALIBRATIONS_ROOT
        / "intrinsic_zero"
        / f"intr_{calibration_name}.xml"
    )

    extrinsic_path = (
        CALIBRATIONS_ROOT
        / "extrinsic"
        / f"extr_{calibration_name}.xml"
    )

    if not intrinsic_path.exists():
        raise FileNotFoundError(
            f"Не найдена внутренняя калибровка: "
            f"{intrinsic_path.resolve()}"
        )

    if not extrinsic_path.exists():
        raise FileNotFoundError(
            f"Не найдена внешняя калибровка: "
            f"{extrinsic_path.resolve()}"
        )

    intrinsic_root = ET.parse(
        intrinsic_path
    ).getroot()

    extrinsic_root = ET.parse(
        extrinsic_path
    ).getroot()

    camera_matrix = read_xml_values(
        intrinsic_root,
        "camera_matrix",
    ).reshape(3, 3)

    rvec = read_xml_values(
        extrinsic_root,
        "rvec",
    ).reshape(3, 1)

    tvec = read_xml_values(
        extrinsic_root,
        "tvec",
    ).reshape(3, 1)

    return camera_matrix, rvec, tvec


def position_id_to_world_m(
    position_id: int,
) -> np.ndarray:
    """
    Перевод positionID в мировые X, Y в метрах.

    Сетка:
        ширина — 480 позиций;
        шаг — 0.025 м;
        начало — (-3, -9) м.
    """
    grid_x = position_id % 480
    grid_y = position_id // 480

    world_x = -3.0 + 0.025 * grid_x
    world_y = -9.0 + 0.025 * grid_y

    return np.array(
        [world_x, world_y],
        dtype=np.float64,
    )


def bbox_foot_point(
    view: dict,
) -> tuple[float, float]:
    xmin = float(view["xmin"])
    ymin = float(view["ymin"])
    xmax = float(view["xmax"])
    ymax = float(view["ymax"])

    if xmin < 0 or ymin < 0:
        raise ValueError("Объект не виден в камере.")

    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Некорректный bounding box.")

    u = (xmin + xmax) / 2.0
    v = ymax

    return u, v


def image_point_to_ground(
    u: float,
    v: float,
    camera_matrix: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    """
    Обратная проекция пикселя на плоскость Z=0.

    Калибровка WILDTRACK использует сантиметры.
    Возвращаем X, Y в метрах.
    """
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    camera_center_world = (
        -rotation_matrix.T @ tvec
    ).reshape(3)

    pixel = np.array(
        [u, v, 1.0],
        dtype=np.float64,
    )

    ray_camera = (
        np.linalg.inv(camera_matrix) @ pixel
    )

    ray_world = (
        rotation_matrix.T @ ray_camera
    )

    if abs(ray_world[2]) < 1e-10:
        raise ValueError(
            "Луч почти параллелен плоскости земли."
        )

    scale = (
        -camera_center_world[2]
        / ray_world[2]
    )

    world_point_cm = (
        camera_center_world
        + scale * ray_world
    )

    return world_point_cm[:2] / 100.0


def world_to_image(
    world_xy_m: np.ndarray,
    camera_matrix: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    """
    Прямая проекция мировой точки Z=0 на изображение.
    """
    world_point_cm = np.array(
        [
            [
                world_xy_m[0] * 100.0,
                world_xy_m[1] * 100.0,
                0.0,
            ]
        ],
        dtype=np.float64,
    )

    image_points, _ = cv2.projectPoints(
        world_point_cm,
        rvec,
        tvec,
        camera_matrix,
        np.zeros((5, 1), dtype=np.float64),
    )

    return image_points.reshape(-1, 2)[0]


def is_visible_view(view: dict) -> bool:
    xmin = float(view["xmin"])
    ymin = float(view["ymin"])
    xmax = float(view["xmax"])
    ymax = float(view["ymax"])

    return (
        xmin >= 0
        and ymin >= 0
        and xmax > xmin
        and ymax > ymin
    )


def camera_to_view_number(
    camera_name: str,
) -> int:
    """
    В WILDTRACK viewNum имеет нумерацию 0..6.
    """
    return int(camera_name.removeprefix("C")) - 1


def calculate_summary(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    summaries: list[dict] = []

    for camera_id, group in rows.groupby(
        "camera_id"
    ):
        summaries.append(
            {
                "camera_id": camera_id,
                "detections": len(group),
                "mean_ground_error_m": (
                    group["ground_error_m"].mean()
                ),
                "median_ground_error_m": (
                    group["ground_error_m"].median()
                ),
                "p90_ground_error_m": (
                    group["ground_error_m"].quantile(0.90)
                ),
                "max_ground_error_m": (
                    group["ground_error_m"].max()
                ),
                "mean_pixel_error": (
                    group["pixel_error"].mean()
                ),
                "median_pixel_error": (
                    group["pixel_error"].median()
                ),
                "p90_pixel_error": (
                    group["pixel_error"].quantile(0.90)
                ),
            }
        )

    return pd.DataFrame(summaries)


def main() -> None:
    args = parse_args()

    annotation_files = sorted(
        ANNOTATIONS_ROOT.glob("*.json")
    )[: args.frames]

    if not annotation_files:
        raise FileNotFoundError(
            f"Аннотации не найдены: "
            f"{ANNOTATIONS_ROOT.resolve()}"
        )

    calibrations = {
        camera: load_calibration(camera)
        for camera in args.cameras
    }

    rows: list[dict] = []

    for annotation_path in annotation_files:
        frame_number = int(annotation_path.stem)

        with annotation_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            annotations = json.load(file)

        for person in annotations:
            person_id = int(person["personID"])
            position_id = int(person["positionID"])

            gt_world_xy = position_id_to_world_m(
                position_id
            )

            views_by_number = {
                int(view["viewNum"]): view
                for view in person.get("views", [])
            }

            for camera in args.cameras:
                view_number = camera_to_view_number(
                    camera
                )

                view = views_by_number.get(view_number)

                if (
                    view is None
                    or not is_visible_view(view)
                ):
                    continue

                camera_matrix, rvec, tvec = (
                    calibrations[camera]
                )

                foot_u, foot_v = bbox_foot_point(
                    view
                )

                predicted_world_xy = (
                    image_point_to_ground(
                        u=foot_u,
                        v=foot_v,
                        camera_matrix=camera_matrix,
                        rvec=rvec,
                        tvec=tvec,
                    )
                )

                projected_pixel = world_to_image(
                    world_xy_m=gt_world_xy,
                    camera_matrix=camera_matrix,
                    rvec=rvec,
                    tvec=tvec,
                )

                ground_error = float(
                    np.linalg.norm(
                        predicted_world_xy
                        - gt_world_xy
                    )
                )

                pixel_error = float(
                    np.linalg.norm(
                        projected_pixel
                        - np.array(
                            [foot_u, foot_v],
                            dtype=np.float64,
                        )
                    )
                )

                rows.append(
                    {
                        "frame": frame_number,
                        "camera_id": camera,
                        "person_id": person_id,
                        "position_id": position_id,
                        "foot_u": foot_u,
                        "foot_v": foot_v,
                        "gt_x_m": gt_world_xy[0],
                        "gt_y_m": gt_world_xy[1],
                        "predicted_x_m": (
                            predicted_world_xy[0]
                        ),
                        "predicted_y_m": (
                            predicted_world_xy[1]
                        ),
                        "ground_error_m": ground_error,
                        "projected_u": projected_pixel[0],
                        "projected_v": projected_pixel[1],
                        "pixel_error": pixel_error,
                    }
                )

    results = pd.DataFrame(rows)

    if results.empty:
        raise RuntimeError(
            "Не удалось получить ни одной проекции."
        )

    summary = calculate_summary(results)

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        args.output / "per_detection.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        args.output / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Обработано кадров: {len(annotation_files)}")
    print(f"Получено проекций: {len(results)}")
    print()
    print(
        summary.round(4).to_string(index=False)
    )
    print()
    print(
        "Результаты:",
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()