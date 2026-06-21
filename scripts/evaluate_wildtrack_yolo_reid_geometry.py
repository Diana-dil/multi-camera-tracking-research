from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TORCHREID_ROOT = PROJECT_ROOT / "external" / "deep-person-reid"

if str(TORCHREID_ROOT) not in sys.path:
    sys.path.insert(0, str(TORCHREID_ROOT))

from torchreid.utils import FeatureExtractor  # noqa: E402


CAMERA_CALIBRATION_NAMES = {
    "C1": "CVLab1",
    "C2": "CVLab2",
    "C3": "CVLab3",
    "C4": "CVLab4",
    "C5": "IDIAP1",
    "C6": "IDIAP2",
    "C7": "IDIAP3",
}

DEFAULT_WEIGHTS = (
    PROJECT_ROOT
    / "models"
    / "reid"
    / (
        "osnet_ain_x1_0_msmt17_256x128_amsgrad_"
        "ep50_lr0.0015_coslr_b64_fb10_softmax_"
        "labsmth_flip_jitter.pth"
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end оценка межкамерного сопоставления: "
            "YOLO-рамки + OSNet + геометрия WILDTRACK."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/wildtrack_C1_C6_yolo_50"),
        help="Папка с detections.csv и crops.",
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
        help="Геометрические пороги в метрах.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
    )
    parser.add_argument(
        "--model-name",
        default="osnet_ain_x1_0",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/wildtrack_yolo_reid_geometry_C1_C6"),
    )
    return parser.parse_args()


def read_xml_values(
    root: ET.Element,
    field_name: str,
) -> np.ndarray:
    node = root.find(field_name)

    if node is None:
        raise KeyError(f"Поле {field_name!r} не найдено в XML.")

    data_node = node.find("data")
    text = data_node.text if data_node is not None else node.text

    if not text:
        raise ValueError(
            f"Поле {field_name!r} не содержит значений."
        )

    return np.array(
        [float(value) for value in text.split()],
        dtype=np.float64,
    )


def load_calibration(
    calibrations_root: Path,
    camera_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def image_point_to_ground(
    u: float,
    v: float,
    camera_matrix: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    camera_center_world = (
        -rotation_matrix.T @ tvec
    ).reshape(3)

    pixel = np.array(
        [u, v, 1.0],
        dtype=np.float64,
    )

    ray_camera = np.linalg.inv(camera_matrix) @ pixel
    ray_world = rotation_matrix.T @ ray_camera

    if abs(ray_world[2]) < 1e-10:
        raise ValueError(
            "Луч почти параллелен плоскости земли."
        )

    scale = -camera_center_world[2] / ray_world[2]
    world_point_cm = camera_center_world + scale * ray_world

    return world_point_cm[:2] / 100.0


def camera_to_view_number(camera_name: str) -> int:
    return int(camera_name.removeprefix("C")) - 1


def is_valid_view(view: dict) -> bool:
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


def load_common_gt_ids(
    annotations_root: Path,
    frame_number: int,
    camera_a: str,
    camera_b: str,
) -> set[int]:
    annotation_path = (
        annotations_root / f"{frame_number:08d}.json"
    )

    if not annotation_path.exists():
        raise FileNotFoundError(annotation_path)

    with annotation_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        annotations = json.load(file)

    visible: dict[str, set[int]] = {
        camera_a: set(),
        camera_b: set(),
    }

    view_numbers = {
        camera_a: camera_to_view_number(camera_a),
        camera_b: camera_to_view_number(camera_b),
    }

    for person in annotations:
        person_id = int(person["personID"])

        views = {
            int(view["viewNum"]): view
            for view in person.get("views", [])
        }

        for camera in (camera_a, camera_b):
            view = views.get(view_numbers[camera])

            if view is not None and is_valid_view(view):
                visible[camera].add(person_id)

    return visible[camera_a] & visible[camera_b]


def resolve_crop_path(
    dataset_root: Path,
    crop_path: str,
) -> Path:
    path = dataset_root / Path(crop_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Не найден crop: {path.resolve()}"
        )

    return path.resolve()


def extract_embeddings(
    detections: pd.DataFrame,
    dataset_root: Path,
    extractor: FeatureExtractor,
    batch_size: int,
) -> np.ndarray:
    image_paths = [
        str(resolve_crop_path(dataset_root, value))
        for value in detections["crop_path"].astype(str)
    ]

    batches: list[torch.Tensor] = []

    for start in range(0, len(image_paths), batch_size):
        end = min(start + batch_size, len(image_paths))

        with torch.no_grad():
            embeddings = extractor(
                image_paths[start:end]
            )

        embeddings = F.normalize(
            embeddings.float(),
            p=2,
            dim=1,
        )

        batches.append(
            embeddings.detach().cpu()
        )

        print(
            f"Embeddings: {end}/{len(image_paths)}",
            end="\r",
        )

    print()

    return torch.cat(
        batches,
        dim=0,
    ).numpy()


def add_ground_coordinates(
    detections: pd.DataFrame,
    calibrations_root: Path,
) -> pd.DataFrame:
    result = detections.copy()

    calibrations = {
        camera: load_calibration(
            calibrations_root,
            camera,
        )
        for camera in result["camera_id"].unique()
    }

    ground_x: list[float] = []
    ground_y: list[float] = []

    for row in result.itertuples():
        camera_matrix, rvec, tvec = calibrations[
            row.camera_id
        ]

        point = image_point_to_ground(
            u=float(row.foot_u),
            v=float(row.foot_v),
            camera_matrix=camera_matrix,
            rvec=rvec,
            tvec=tvec,
        )

        ground_x.append(float(point[0]))
        ground_y.append(float(point[1]))

    result["ground_x_m"] = ground_x
    result["ground_y_m"] = ground_y

    return result


def matched_detection_map(
    camera_detections: pd.DataFrame,
) -> dict[int, pd.Series]:
    matched = camera_detections[
        (camera_detections["matched_gt"] == 1)
        & (camera_detections["gt_person_id"] >= 0)
    ]

    result: dict[int, pd.Series] = {}

    for _, row in matched.iterrows():
        person_id = int(row["gt_person_id"])

        current = result.get(person_id)

        if (
            current is None
            or float(row["match_iou"])
            > float(current["match_iou"])
        ):
            result[person_id] = row

    return result


def evaluate_direction(
    detections: pd.DataFrame,
    embeddings: np.ndarray,
    annotations_root: Path,
    query_camera: str,
    gallery_camera: str,
    thresholds: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []

    frame_numbers = sorted(
        detections["frame"].astype(int).unique()
    )

    for frame_number in frame_numbers:
        common_gt_ids = load_common_gt_ids(
            annotations_root=annotations_root,
            frame_number=int(frame_number),
            camera_a=query_camera,
            camera_b=gallery_camera,
        )

        frame_detections = detections[
            detections["frame"] == frame_number
        ]

        query_group = frame_detections[
            frame_detections["camera_id"]
            == query_camera
        ]

        gallery_group = frame_detections[
            frame_detections["camera_id"]
            == gallery_camera
        ]

        query_map = matched_detection_map(query_group)
        gallery_map = matched_detection_map(gallery_group)

        gallery_indices = (
            gallery_group["embedding_index"]
            .astype(int)
            .to_numpy()
        )

        gallery_ids = (
            gallery_group["gt_person_id"]
            .astype(int)
            .to_numpy()
        )

        gallery_xy = gallery_group[
            ["ground_x_m", "ground_y_m"]
        ].to_numpy(dtype=np.float64)

        for person_id in sorted(common_gt_ids):
            query_row = query_map.get(person_id)
            gallery_row = gallery_map.get(person_id)

            base = {
                "frame": int(frame_number),
                "query_camera": query_camera,
                "gallery_camera": gallery_camera,
                "gt_person_id": int(person_id),
                "query_present": int(query_row is not None),
                "gallery_present": int(
                    gallery_row is not None
                ),
                "both_detected": int(
                    query_row is not None
                    and gallery_row is not None
                ),
                "gallery_size": int(
                    len(gallery_group)
                ),
                "appearance_correct": 0,
                "geometry_correct": 0,
                "appearance_prediction_gt": -1,
                "geometry_prediction_gt": -1,
                "positive_similarity": np.nan,
                "positive_distance_m": np.nan,
            }

            for threshold in thresholds:
                key = f"gate_{threshold:.2f}"
                base[f"{key}_correct"] = 0
                base[f"{key}_prediction_gt"] = -1
                base[f"{key}_candidates"] = 0
                base[f"{key}_fallback"] = 0

            if query_row is None:
                base["status"] = "query_missed"
                rows.append(base)
                continue

            if gallery_row is None:
                base["status"] = "gallery_missed"
                rows.append(base)
                continue

            if gallery_group.empty:
                base["status"] = "gallery_empty"
                rows.append(base)
                continue

            base["status"] = "both_detected"

            query_embedding_index = int(
                query_row["embedding_index"]
            )

            query_embedding = embeddings[
                query_embedding_index
            ]

            gallery_embeddings = embeddings[
                gallery_indices
            ]

            similarities = (
                gallery_embeddings @ query_embedding
            )

            query_xy = np.array(
                [
                    float(query_row["ground_x_m"]),
                    float(query_row["ground_y_m"]),
                ],
                dtype=np.float64,
            )

            distances = np.linalg.norm(
                gallery_xy - query_xy,
                axis=1,
            )

            gallery_global_indices = (
                gallery_group.index.to_numpy()
            )

            true_global_index = int(
                gallery_row.name
            )

            true_positions = np.where(
                gallery_global_indices
                == true_global_index
            )[0]

            if len(true_positions) != 1:
                raise RuntimeError(
                    "Не удалось однозначно найти "
                    "правильную gallery-детекцию."
                )

            true_position = int(
                true_positions[0]
            )

            appearance_position = int(
                np.argmax(similarities)
            )

            geometry_position = int(
                np.argmin(distances)
            )

            appearance_prediction = int(
                gallery_ids[appearance_position]
            )

            geometry_prediction = int(
                gallery_ids[geometry_position]
            )

            base["appearance_prediction_gt"] = (
                appearance_prediction
            )
            base["geometry_prediction_gt"] = (
                geometry_prediction
            )
            base["appearance_correct"] = int(
                appearance_prediction == person_id
            )
            base["geometry_correct"] = int(
                geometry_prediction == person_id
            )
            base["positive_similarity"] = float(
                similarities[true_position]
            )
            base["positive_distance_m"] = float(
                distances[true_position]
            )

            for threshold in thresholds:
                key = f"gate_{threshold:.2f}"

                candidates = np.where(
                    distances <= threshold
                )[0]

                if len(candidates) == 0:
                    selected_position = (
                        appearance_position
                    )
                    fallback = 1
                else:
                    local_best = int(
                        np.argmax(
                            similarities[candidates]
                        )
                    )
                    selected_position = int(
                        candidates[local_best]
                    )
                    fallback = 0

                prediction = int(
                    gallery_ids[selected_position]
                )

                base[f"{key}_prediction_gt"] = (
                    prediction
                )
                base[f"{key}_correct"] = int(
                    prediction == person_id
                )
                base[f"{key}_candidates"] = int(
                    len(candidates)
                )
                base[f"{key}_fallback"] = (
                    fallback
                )

            rows.append(base)

    per_query = pd.DataFrame(rows)

    if per_query.empty:
        raise RuntimeError(
            f"Нет GT-запросов для "
            f"{query_camera} -> {gallery_camera}"
        )

    both = per_query[
        per_query["both_detected"] == 1
    ]

    summary_rows: list[dict] = []

    method_specs = [
        (
            "appearance_only",
            np.nan,
            "appearance_correct",
            None,
            None,
        ),
        (
            "geometry_only",
            np.nan,
            "geometry_correct",
            None,
            None,
        ),
    ]

    for threshold in thresholds:
        key = f"gate_{threshold:.2f}"

        method_specs.append(
            (
                "appearance_plus_geometry_gate",
                threshold,
                f"{key}_correct",
                f"{key}_candidates",
                f"{key}_fallback",
            )
        )

    total_gt_queries = len(per_query)
    both_detected = int(
        per_query["both_detected"].sum()
    )
    query_detected = int(
        per_query["query_present"].sum()
    )
    gallery_detected = int(
        per_query["gallery_present"].sum()
    )

    for (
        method,
        threshold,
        correct_column,
        candidates_column,
        fallback_column,
    ) in method_specs:
        correct = int(
            per_query[correct_column].sum()
        )

        summary_rows.append(
            {
                "query_camera": query_camera,
                "gallery_camera": gallery_camera,
                "method": method,
                "threshold_m": threshold,
                "total_common_gt_queries": (
                    total_gt_queries
                ),
                "query_detected": query_detected,
                "gallery_detected": gallery_detected,
                "both_detected": both_detected,
                "pair_detection_coverage": (
                    both_detected
                    / total_gt_queries
                    if total_gt_queries
                    else 0.0
                ),
                "correct_associations": correct,
                "conditional_rank1": (
                    correct / both_detected
                    if both_detected
                    else 0.0
                ),
                "end_to_end_correct_rate": (
                    correct / total_gt_queries
                    if total_gt_queries
                    else 0.0
                ),
                "mean_candidates": (
                    float(
                        both[
                            candidates_column
                        ].mean()
                    )
                    if candidates_column
                    and not both.empty
                    else np.nan
                ),
                "fallback_rate": (
                    float(
                        both[
                            fallback_column
                        ].mean()
                    )
                    if fallback_column
                    and not both.empty
                    else 0.0
                ),
            }
        )

    return per_query, pd.DataFrame(summary_rows)


def create_overall_summary(
    direction_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    grouping_columns = [
        "method",
        "threshold_m",
    ]

    for keys, group in direction_summary.groupby(
        grouping_columns,
        dropna=False,
        sort=False,
    ):
        method, threshold = keys

        total_queries = int(
            group["total_common_gt_queries"].sum()
        )
        both_detected = int(
            group["both_detected"].sum()
        )
        correct = int(
            group["correct_associations"].sum()
        )

        candidate_weights = group[
            "both_detected"
        ].to_numpy(dtype=np.float64)

        if (
            group["mean_candidates"].notna().any()
            and candidate_weights.sum() > 0
        ):
            mean_candidates = float(
                np.nansum(
                    group["mean_candidates"].to_numpy()
                    * candidate_weights
                )
                / candidate_weights.sum()
            )
        else:
            mean_candidates = np.nan

        if candidate_weights.sum() > 0:
            fallback_rate = float(
                np.nansum(
                    group["fallback_rate"].to_numpy()
                    * candidate_weights
                )
                / candidate_weights.sum()
            )
        else:
            fallback_rate = 0.0

        rows.append(
            {
                "method": method,
                "threshold_m": threshold,
                "total_common_gt_queries": total_queries,
                "both_detected": both_detected,
                "pair_detection_coverage": (
                    both_detected / total_queries
                    if total_queries
                    else 0.0
                ),
                "correct_associations": correct,
                "conditional_rank1": (
                    correct / both_detected
                    if both_detected
                    else 0.0
                ),
                "end_to_end_correct_rate": (
                    correct / total_queries
                    if total_queries
                    else 0.0
                ),
                "mean_candidates": mean_candidates,
                "fallback_rate": fallback_rate,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "end_to_end_correct_rate",
                "conditional_rank1",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset.resolve()
    detections_path = (
        dataset_root / "detections.csv"
    )

    if not detections_path.exists():
        raise FileNotFoundError(
            detections_path
        )

    weights_path = args.weights.resolve()

    if not weights_path.exists():
        raise FileNotFoundError(
            weights_path
        )

    thresholds = [
        float(value.strip())
        for value in args.thresholds.split(",")
        if value.strip()
    ]

    detections = (
        pd.read_csv(detections_path)
        .reset_index(drop=True)
    )

    required_columns = {
        "frame",
        "camera_id",
        "foot_u",
        "foot_v",
        "matched_gt",
        "gt_person_id",
        "match_iou",
        "crop_path",
    }

    missing = (
        required_columns
        - set(detections.columns)
    )

    if missing:
        raise ValueError(
            "В detections.csv отсутствуют поля: "
            f"{sorted(missing)}"
        )

    detections["frame"] = (
        detections["frame"].astype(int)
    )
    detections["gt_person_id"] = (
        detections["gt_person_id"].astype(int)
    )
    detections["matched_gt"] = (
        detections["matched_gt"].astype(int)
    )
    detections["embedding_index"] = np.arange(
        len(detections)
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(f"Detections: {len(detections)}")
    print(f"Model: {args.model_name}")

    extractor = FeatureExtractor(
        model_name=args.model_name,
        model_path=str(weights_path),
        device=device,
    )

    embeddings = extract_embeddings(
        detections=detections,
        dataset_root=dataset_root,
        extractor=extractor,
        batch_size=args.batch_size,
    )

    detections = add_ground_coordinates(
        detections=detections,
        calibrations_root=(
            args.calibrations.resolve()
        ),
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        args.output / "embeddings.npy",
        embeddings,
    )

    detections.to_csv(
        args.output
        / "detections_with_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_per_query: list[pd.DataFrame] = []
    all_direction_summary: list[pd.DataFrame] = []

    for query_camera, gallery_camera in [
        (args.camera_a, args.camera_b),
        (args.camera_b, args.camera_a),
    ]:
        per_query, summary = (
            evaluate_direction(
                detections=detections,
                embeddings=embeddings,
                annotations_root=(
                    args.annotations.resolve()
                ),
                query_camera=query_camera,
                gallery_camera=gallery_camera,
                thresholds=thresholds,
            )
        )

        all_per_query.append(per_query)
        all_direction_summary.append(summary)

    per_query_results = pd.concat(
        all_per_query,
        ignore_index=True,
    )

    direction_summary = pd.concat(
        all_direction_summary,
        ignore_index=True,
    )

    overall_summary = create_overall_summary(
        direction_summary
    )

    per_query_results.to_csv(
        args.output / "per_query_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    direction_summary.to_csv(
        args.output / "direction_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall_summary.to_csv(
        args.output / "overall_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    both = per_query_results[
        per_query_results["both_detected"] == 1
    ]

    distance_summary = {
        "both_detected_queries": int(len(both)),
        "positive_distance_mean_m": (
            float(
                both[
                    "positive_distance_m"
                ].mean()
            )
            if not both.empty
            else None
        ),
        "positive_distance_median_m": (
            float(
                both[
                    "positive_distance_m"
                ].median()
            )
            if not both.empty
            else None
        ),
        "positive_distance_p90_m": (
            float(
                both[
                    "positive_distance_m"
                ].quantile(0.90)
            )
            if not both.empty
            else None
        ),
    }

    with (
        args.output / "distance_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            distance_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("Итог по двум направлениям:")
    print(
        overall_summary.round(4).to_string(
            index=False
        )
    )
    print()
    print(
        "Результаты:",
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
