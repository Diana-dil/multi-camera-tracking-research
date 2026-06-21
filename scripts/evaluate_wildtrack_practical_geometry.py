from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


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
            "Оценка межкамерного ReID с геометрией, вычисленной "
            "по калибровке камер WILDTRACK."
        )
    )
    parser.add_argument(
        "--reid-results",
        type=Path,
        default=Path("results/wildtrack_reid_C1_C6"),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/raw/WILDTRACK/annotations_positions"),
    )
    parser.add_argument(
        "--calibrations",
        type=Path,
        default=Path("data/raw/WILDTRACK/calibrations"),
    )
    parser.add_argument("--camera-a", default="C1")
    parser.add_argument("--camera-b", default="C6")
    parser.add_argument(
        "--thresholds",
        default="0.25,0.50,0.75,1.00,1.50",
        help="Пороговые расстояния на плоскости земли, метры.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/wildtrack_practical_geometry_C1_C6"),
    )
    return parser.parse_args()


def read_xml_values(root: ET.Element, field_name: str) -> np.ndarray:
    node = root.find(field_name)
    if node is None:
        raise KeyError(f"Поле {field_name!r} не найдено в XML.")

    data_node = node.find("data")
    text = data_node.text if data_node is not None else node.text

    if not text:
        raise ValueError(f"Поле {field_name!r} не содержит значений.")

    return np.array(
        [float(value) for value in text.split()],
        dtype=np.float64,
    )


def load_calibration(
    calibrations_root: Path,
    camera_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if camera_name not in CAMERA_CALIBRATION_NAMES:
        raise ValueError(f"Неизвестная камера: {camera_name}")

    calibration_name = CAMERA_CALIBRATION_NAMES[camera_name]

    intrinsic_path = (
        calibrations_root
        / "intrinsic_zero"
        / f"intr_{calibration_name}.xml"
    )
    extrinsic_path = (
        calibrations_root
        / "extrinsic"
        / f"extr_{calibration_name}.xml"
    )

    if not intrinsic_path.exists():
        raise FileNotFoundError(intrinsic_path)
    if not extrinsic_path.exists():
        raise FileNotFoundError(extrinsic_path)

    intrinsic_root = ET.parse(intrinsic_path).getroot()
    extrinsic_root = ET.parse(extrinsic_path).getroot()

    camera_matrix = read_xml_values(
        intrinsic_root, "camera_matrix"
    ).reshape(3, 3)
    rvec = read_xml_values(
        extrinsic_root, "rvec"
    ).reshape(3, 1)
    tvec = read_xml_values(
        extrinsic_root, "tvec"
    ).reshape(3, 1)

    return camera_matrix, rvec, tvec


def image_point_to_ground(
    u: float,
    v: float,
    camera_matrix: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    """Обратная проекция пикселя на плоскость Z=0; результат в метрах."""
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    camera_center_world = (
        -rotation_matrix.T @ tvec
    ).reshape(3)

    pixel = np.array([u, v, 1.0], dtype=np.float64)
    ray_camera = np.linalg.inv(camera_matrix) @ pixel
    ray_world = rotation_matrix.T @ ray_camera

    if abs(ray_world[2]) < 1e-10:
        raise ValueError("Луч почти параллелен плоскости земли.")

    scale = -camera_center_world[2] / ray_world[2]
    world_point_cm = camera_center_world + scale * ray_world

    return world_point_cm[:2] / 100.0


def camera_to_view_number(camera_name: str) -> int:
    return int(camera_name.removeprefix("C")) - 1


def valid_view(view: dict) -> bool:
    x1 = float(view["xmin"])
    y1 = float(view["ymin"])
    x2 = float(view["xmax"])
    y2 = float(view["ymax"])
    return x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1


def build_ground_map(
    metadata: pd.DataFrame,
    annotations_root: Path,
    calibrations_root: Path,
    cameras: list[str],
) -> dict[tuple[int, str, int], np.ndarray]:
    calibrations = {
        camera: load_calibration(calibrations_root, camera)
        for camera in cameras
    }

    frame_numbers = sorted(
        {int(value) for value in metadata["frame"].tolist()}
    )

    result: dict[tuple[int, str, int], np.ndarray] = {}

    for frame_number in frame_numbers:
        annotation_path = annotations_root / f"{frame_number:08d}.json"

        if not annotation_path.exists():
            raise FileNotFoundError(annotation_path)

        with annotation_path.open("r", encoding="utf-8") as file:
            annotations = json.load(file)

        for person in annotations:
            person_id = int(person["personID"])
            views = {
                int(view["viewNum"]): view
                for view in person.get("views", [])
            }

            for camera in cameras:
                view = views.get(camera_to_view_number(camera))

                if view is None or not valid_view(view):
                    continue

                u = (
                    float(view["xmin"]) + float(view["xmax"])
                ) / 2.0
                v = float(view["ymax"])

                camera_matrix, rvec, tvec = calibrations[camera]

                result[(frame_number, camera, person_id)] = (
                    image_point_to_ground(
                        u=u,
                        v=v,
                        camera_matrix=camera_matrix,
                        rvec=rvec,
                        tvec=tvec,
                    )
                )

    return result


def add_ground_coordinates(
    metadata: pd.DataFrame,
    ground_map: dict[tuple[int, str, int], np.ndarray],
) -> pd.DataFrame:
    result = metadata.copy()
    result["frame"] = result["frame"].astype(int)
    result["person_id"] = result["person_id"].astype(int)

    xs: list[float] = []
    ys: list[float] = []

    missing: list[tuple[int, str, int]] = []

    for row in result.itertuples():
        key = (
            int(row.frame),
            str(row.camera_id),
            int(row.person_id),
        )
        point = ground_map.get(key)

        if point is None:
            missing.append(key)
            xs.append(np.nan)
            ys.append(np.nan)
        else:
            xs.append(float(point[0]))
            ys.append(float(point[1]))

    result["ground_x_m"] = xs
    result["ground_y_m"] = ys

    if missing:
        raise ValueError(
            "Не найдены координаты для части строк. Примеры: "
            f"{missing[:10]}"
        )

    return result


def evaluate_direction(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    query_camera: str,
    gallery_camera: str,
    thresholds: list[float],
) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []

    for frame, frame_group in metadata.groupby("frame", sort=True):
        query_group = frame_group[
            frame_group["camera_id"] == query_camera
        ]
        gallery_group = frame_group[
            frame_group["camera_id"] == gallery_camera
        ]

        if query_group.empty or gallery_group.empty:
            continue

        query_indices = (
            query_group["embedding_index"].astype(int).to_numpy()
        )
        gallery_indices = (
            gallery_group["embedding_index"].astype(int).to_numpy()
        )

        query_embeddings = embeddings[query_indices]
        gallery_embeddings = embeddings[gallery_indices]
        similarity_matrix = query_embeddings @ gallery_embeddings.T

        query_xy = query_group[
            ["ground_x_m", "ground_y_m"]
        ].to_numpy(dtype=np.float64)
        gallery_xy = gallery_group[
            ["ground_x_m", "ground_y_m"]
        ].to_numpy(dtype=np.float64)

        distance_matrix = np.linalg.norm(
            query_xy[:, None, :] - gallery_xy[None, :, :],
            axis=2,
        )

        gallery_ids = (
            gallery_group["person_id"].astype(int).to_numpy()
        )

        for query_position, (_, query_row) in enumerate(
            query_group.iterrows()
        ):
            query_person_id = int(query_row["person_id"])

            if not np.any(gallery_ids == query_person_id):
                continue

            similarities = similarity_matrix[query_position]
            distances = distance_matrix[query_position]

            appearance_index = int(np.argmax(similarities))
            geometry_index = int(np.argmin(distances))

            row = {
                "frame": int(frame),
                "query_camera": query_camera,
                "gallery_camera": gallery_camera,
                "query_person_id": query_person_id,
                "gallery_size": int(len(gallery_group)),
                "appearance_prediction": int(
                    gallery_ids[appearance_index]
                ),
                "appearance_correct": int(
                    gallery_ids[appearance_index] == query_person_id
                ),
                "geometry_prediction": int(
                    gallery_ids[geometry_index]
                ),
                "geometry_correct": int(
                    gallery_ids[geometry_index] == query_person_id
                ),
                "nearest_distance_m": float(
                    distances[geometry_index]
                ),
            }

            positive_positions = np.where(
                gallery_ids == query_person_id
            )[0]
            row["positive_distance_m"] = float(
                distances[positive_positions[0]]
            )

            negative_mask = gallery_ids != query_person_id
            row["nearest_negative_distance_m"] = float(
                distances[negative_mask].min()
            ) if np.any(negative_mask) else np.nan

            for threshold in thresholds:
                key = f"gate_{threshold:.2f}"
                candidate_positions = np.where(
                    distances <= threshold
                )[0]

                if len(candidate_positions) == 0:
                    selected_index = appearance_index
                    fallback = 1
                else:
                    local_best = int(
                        np.argmax(similarities[candidate_positions])
                    )
                    selected_index = int(
                        candidate_positions[local_best]
                    )
                    fallback = 0

                row[f"{key}_prediction"] = int(
                    gallery_ids[selected_index]
                )
                row[f"{key}_correct"] = int(
                    gallery_ids[selected_index] == query_person_id
                )
                row[f"{key}_candidates"] = int(
                    len(candidate_positions)
                )
                row[f"{key}_fallback"] = fallback

            rows.append(row)

    results = pd.DataFrame(rows)

    if results.empty:
        raise RuntimeError(
            f"Нет запросов для {query_camera} -> {gallery_camera}"
        )

    summaries: list[dict] = [
        {
            "query_camera": query_camera,
            "gallery_camera": gallery_camera,
            "method": "appearance_only",
            "threshold_m": np.nan,
            "queries": len(results),
            "rank1": float(results["appearance_correct"].mean()),
            "mean_candidates": float(results["gallery_size"].mean()),
            "fallback_rate": 0.0,
        },
        {
            "query_camera": query_camera,
            "gallery_camera": gallery_camera,
            "method": "geometry_only",
            "threshold_m": np.nan,
            "queries": len(results),
            "rank1": float(results["geometry_correct"].mean()),
            "mean_candidates": 1.0,
            "fallback_rate": 0.0,
        },
    ]

    for threshold in thresholds:
        key = f"gate_{threshold:.2f}"
        summaries.append(
            {
                "query_camera": query_camera,
                "gallery_camera": gallery_camera,
                "method": "appearance_plus_geometry_gate",
                "threshold_m": threshold,
                "queries": len(results),
                "rank1": float(
                    results[f"{key}_correct"].mean()
                ),
                "mean_candidates": float(
                    results[f"{key}_candidates"].mean()
                ),
                "fallback_rate": float(
                    results[f"{key}_fallback"].mean()
                ),
            }
        )

    return results, summaries


def main() -> None:
    args = parse_args()

    thresholds = [
        float(value.strip())
        for value in args.thresholds.split(",")
        if value.strip()
    ]

    reid_root = args.reid_results.resolve()
    metadata_path = reid_root / "embedding_metadata.csv"
    embeddings_path = reid_root / "embeddings.npy"

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(embeddings_path)

    metadata = pd.read_csv(metadata_path)
    embeddings = np.load(embeddings_path)

    if "embedding_index" not in metadata.columns:
        metadata["embedding_index"] = np.arange(len(metadata))

    if len(metadata) != len(embeddings):
        raise ValueError(
            f"metadata={len(metadata)}, embeddings={len(embeddings)}"
        )

    cameras = [args.camera_a, args.camera_b]

    ground_map = build_ground_map(
        metadata=metadata,
        annotations_root=args.annotations.resolve(),
        calibrations_root=args.calibrations.resolve(),
        cameras=cameras,
    )

    metadata = add_ground_coordinates(
        metadata=metadata,
        ground_map=ground_map,
    )

    args.output.mkdir(parents=True, exist_ok=True)

    metadata.to_csv(
        args.output / "metadata_with_projected_ground.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_results: list[pd.DataFrame] = []
    all_summaries: list[dict] = []

    for query_camera, gallery_camera in [
        (args.camera_a, args.camera_b),
        (args.camera_b, args.camera_a),
    ]:
        results, summaries = evaluate_direction(
            metadata=metadata,
            embeddings=embeddings,
            query_camera=query_camera,
            gallery_camera=gallery_camera,
            thresholds=thresholds,
        )

        all_results.append(results)
        all_summaries.extend(summaries)

    per_query = pd.concat(all_results, ignore_index=True)
    summary = pd.DataFrame(all_summaries)

    per_query.to_csv(
        args.output / "per_query_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        args.output / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall = (
        summary.groupby(["method", "threshold_m"], dropna=False)
        .agg(
            rank1=("rank1", "mean"),
            mean_candidates=("mean_candidates", "mean"),
            fallback_rate=("fallback_rate", "mean"),
        )
        .reset_index()
        .sort_values(
            ["rank1", "method"],
            ascending=[False, True],
        )
    )

    overall.to_csv(
        args.output / "overall_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    distance_summary = {
        "positive_distance_mean_m": float(
            per_query["positive_distance_m"].mean()
        ),
        "positive_distance_median_m": float(
            per_query["positive_distance_m"].median()
        ),
        "positive_distance_p90_m": float(
            per_query["positive_distance_m"].quantile(0.90)
        ),
        "nearest_negative_distance_median_m": float(
            per_query["nearest_negative_distance_m"].median()
        ),
        "nearest_negative_distance_p10_m": float(
            per_query["nearest_negative_distance_m"].quantile(0.10)
        ),
    }

    with (args.output / "distance_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            distance_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("Итог по двум направлениям:")
    print(overall.round(4).to_string(index=False))
    print()
    print("Распределение расстояний:")
    for key, value in distance_summary.items():
        print(f"  {key}: {value:.4f}")
    print()
    print(f"Результаты: {args.output.resolve()}")


if __name__ == "__main__":
    main()
