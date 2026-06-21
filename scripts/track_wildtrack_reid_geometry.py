from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TORCHREID_ROOT = PROJECT_ROOT / "external" / "deep-person-reid"

if str(TORCHREID_ROOT) not in sys.path:
    sys.path.insert(0, str(TORCHREID_ROOT))

from torchreid.utils import FeatureExtractor  # noqa: E402


INF_COST = 1_000_000.0

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


@dataclass
class ActiveTrack:
    track_id: int
    last_frame_index: int
    last_x_m: float
    last_y_m: float
    embedding: np.ndarray
    observations: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Собственный трекер WILDTRACK на основе "
            "OSNet + геометрии + венгерского назначения."
        )
    )

    parser.add_argument(
        "--observations",
        type=Path,
        required=True,
        help=(
            "observations.csv после track_wildtrack_sequence.py. "
            "Исходные track_id используются только для сравнения."
        ),
    )

    parser.add_argument(
        "--crops-root",
        type=Path,
        required=True,
        help=(
            "Корневая папка запуска, относительно которой "
            "записаны crop_path в observations.csv."
        ),
    )

    parser.add_argument(
        "--calibrations",
        type=Path,
        default=Path("data/raw/WILDTRACK/calibrations"),
    )

    parser.add_argument(
        "--camera",
        required=True,
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
        "--alpha",
        type=float,
        default=0.75,
        help="Вес appearance-cost; вес геометрии равен 1-alpha.",
    )

    parser.add_argument(
        "--max-distance-per-frame",
        type=float,
        default=2.0,
        help=(
            "Допустимое перемещение на один шаг последовательности, "
            "метры. Для gap>1 порог масштабируется линейно."
        ),
    )

    parser.add_argument(
        "--max-gap",
        type=int,
        default=3,
        help=(
            "Максимальная разница frame_index между последним "
            "наблюдением трека и новой детекцией."
        ),
    )

    parser.add_argument(
        "--acceptance-cost",
        type=float,
        default=0.50,
        help=(
            "Максимальная fused-cost для принятия соответствия."
        ),
    )

    parser.add_argument(
        "--embedding-momentum",
        type=float,
        default=0.80,
        help=(
            "Вес накопленного embedding при обновлении трека."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
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
    if camera_name not in CAMERA_CALIBRATION_NAMES:
        raise ValueError(
            f"Неизвестная камера: {camera_name}"
        )

    calibration_name = CAMERA_CALIBRATION_NAMES[
        camera_name
    ]

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

    world_point_cm = (
        camera_center_world
        + scale * ray_world
    )

    return world_point_cm[:2] / 100.0


def resolve_crop_path(
    crops_root: Path,
    relative_path: str,
) -> Path:
    if not relative_path or relative_path.lower() == "nan":
        raise ValueError(
            "В observations.csv найден пустой crop_path."
        )

    path = crops_root / Path(relative_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Не найден crop: {path.resolve()}"
        )

    return path.resolve()


def extract_embeddings(
    observations: pd.DataFrame,
    crops_root: Path,
    extractor: FeatureExtractor,
    batch_size: int,
) -> np.ndarray:
    image_paths = [
        str(
            resolve_crop_path(
                crops_root,
                str(value),
            )
        )
        for value in observations["crop_path"]
    ]

    batches: list[torch.Tensor] = []

    for start in range(
        0,
        len(image_paths),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(image_paths),
        )

        with torch.no_grad():
            batch_embeddings = extractor(
                image_paths[start:end]
            )

        batch_embeddings = F.normalize(
            batch_embeddings.float(),
            p=2,
            dim=1,
        )

        batches.append(
            batch_embeddings.detach().cpu()
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


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))

    if norm <= 1e-12:
        return vector

    return vector / norm


def solve_with_unmatched(
    real_cost: np.ndarray,
    acceptance_cost: float,
) -> list[tuple[int, int, float]]:
    track_count, detection_count = real_cost.shape

    if track_count == 0 or detection_count == 0:
        return []

    total_size = track_count + detection_count

    augmented = np.full(
        (total_size, total_size),
        INF_COST,
        dtype=np.float64,
    )

    augmented[
        :track_count,
        :detection_count,
    ] = real_cost

    dummy_cost = acceptance_cost / 2.0

    for track_index in range(track_count):
        augmented[
            track_index,
            detection_count + track_index,
        ] = dummy_cost

    for detection_index in range(detection_count):
        augmented[
            track_count + detection_index,
            detection_index,
        ] = dummy_cost

    augmented[
        track_count:,
        detection_count:,
    ] = 0.0

    row_indices, column_indices = (
        linear_sum_assignment(augmented)
    )

    accepted: list[tuple[int, int, float]] = []

    for row_index, column_index in zip(
        row_indices,
        column_indices,
    ):
        if (
            row_index < track_count
            and column_index < detection_count
        ):
            cost = float(
                real_cost[
                    row_index,
                    column_index,
                ]
            )

            if (
                cost < INF_COST / 2.0
                and cost <= acceptance_cost
            ):
                accepted.append(
                    (
                        int(row_index),
                        int(column_index),
                        cost,
                    )
                )

    return accepted


def build_tracks_summary(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for track_id, group in observations.groupby(
        "track_id",
        sort=True,
    ):
        ordered = group.sort_values(
            "frame_index"
        )

        rows.append(
            {
                "camera_id": str(
                    ordered.iloc[0]["camera_id"]
                ),
                "track_id": int(track_id),
                "first_frame_index": int(
                    ordered["frame_index"].min()
                ),
                "last_frame_index": int(
                    ordered["frame_index"].max()
                ),
                "first_frame_name": str(
                    ordered.iloc[0]["frame_name"]
                ),
                "last_frame_name": str(
                    ordered.iloc[-1]["frame_name"]
                ),
                "observations": int(
                    len(ordered)
                ),
                "duration_frames": int(
                    ordered["frame_index"].max()
                    - ordered["frame_index"].min()
                    + 1
                ),
                "mean_confidence": float(
                    ordered["confidence"].mean()
                ),
                "mean_bbox_width": float(
                    ordered["bbox_width"].mean()
                ),
                "mean_bbox_height": float(
                    ordered["bbox_height"].mean()
                ),
                "source_track_ids": int(
                    ordered[
                        "source_track_id"
                    ].nunique()
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha должен быть в диапазоне [0, 1].")

    if not 0.0 <= args.embedding_momentum < 1.0:
        raise ValueError(
            "--embedding-momentum должен быть в диапазоне [0, 1)."
        )

    observations_path = args.observations.resolve()
    crops_root = args.crops_root.resolve()
    weights_path = args.weights.resolve()

    if not observations_path.exists():
        raise FileNotFoundError(observations_path)

    if not crops_root.exists():
        raise FileNotFoundError(crops_root)

    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    observations = (
        pd.read_csv(observations_path)
        .reset_index(drop=True)
    )

    required_columns = {
        "camera_id",
        "frame_index",
        "frame_name",
        "track_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "bbox_width",
        "bbox_height",
        "foot_u",
        "foot_v",
        "crop_path",
    }

    missing_columns = (
        required_columns
        - set(observations.columns)
    )

    if missing_columns:
        raise ValueError(
            "В observations.csv отсутствуют поля: "
            f"{sorted(missing_columns)}"
        )

    observations = observations[
        observations["camera_id"] == args.camera
    ].copy()

    if observations.empty:
        raise ValueError(
            f"В observations.csv нет камеры {args.camera}."
        )

    observations["frame_index"] = (
        observations["frame_index"].astype(int)
    )

    observations["source_track_id"] = (
        observations["track_id"].astype(int)
    )

    observations["embedding_index"] = np.arange(
        len(observations)
    )

    camera_matrix, rvec, tvec = load_calibration(
        calibrations_root=args.calibrations.resolve(),
        camera_name=args.camera,
    )

    ground_points = [
        image_point_to_ground(
            u=float(row.foot_u),
            v=float(row.foot_v),
            camera_matrix=camera_matrix,
            rvec=rvec,
            tvec=tvec,
        )
        for row in observations.itertuples()
    ]

    observations["ground_x_m"] = [
        float(point[0])
        for point in ground_points
    ]

    observations["ground_y_m"] = [
        float(point[1])
        for point in ground_points
    ]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(f"Camera: {args.camera}")
    print(f"Detections: {len(observations)}")
    print(f"Model: {args.model_name}")

    extractor = FeatureExtractor(
        model_name=args.model_name,
        model_path=str(weights_path),
        device=device,
    )

    embeddings = extract_embeddings(
        observations=observations,
        crops_root=crops_root,
        extractor=extractor,
        batch_size=args.batch_size,
    )

    frame_indices = sorted(
        observations["frame_index"].unique()
    )

    active_tracks: dict[int, ActiveTrack] = {}
    next_track_id = 1

    assigned_track_ids = np.full(
        len(observations),
        -1,
        dtype=np.int64,
    )

    assignment_costs = np.full(
        len(observations),
        np.nan,
        dtype=np.float64,
    )

    assignment_similarities = np.full(
        len(observations),
        np.nan,
        dtype=np.float64,
    )

    assignment_distances = np.full(
        len(observations),
        np.nan,
        dtype=np.float64,
    )

    assignment_gaps = np.full(
        len(observations),
        0,
        dtype=np.int64,
    )

    for frame_number, frame_index in enumerate(
        frame_indices,
        start=1,
    ):
        frame_rows = observations[
            observations["frame_index"] == frame_index
        ]

        detection_indices = (
            frame_rows.index.to_numpy()
        )

        detection_embeddings = embeddings[
            detection_indices
        ]

        detection_xy = frame_rows[
            ["ground_x_m", "ground_y_m"]
        ].to_numpy(dtype=np.float64)

        active_items = [
            (track_id, track)
            for track_id, track in active_tracks.items()
            if (
                frame_index
                - track.last_frame_index
            ) <= args.max_gap
        ]

        active_tracks = {
            track_id: track
            for track_id, track in active_items
        }

        active_items = list(
            active_tracks.items()
        )

        if active_items and len(detection_indices) > 0:
            track_embeddings = np.stack(
                [
                    track.embedding
                    for _, track in active_items
                ],
                axis=0,
            )

            track_xy = np.array(
                [
                    [
                        track.last_x_m,
                        track.last_y_m,
                    ]
                    for _, track in active_items
                ],
                dtype=np.float64,
            )

            gaps = np.array(
                [
                    frame_index
                    - track.last_frame_index
                    for _, track in active_items
                ],
                dtype=np.float64,
            )

            similarities = (
                track_embeddings
                @ detection_embeddings.T
            )

            appearance_cost = 1.0 - similarities

            distances = np.linalg.norm(
                track_xy[:, None, :]
                - detection_xy[None, :, :],
                axis=2,
            )

            distance_thresholds = (
                args.max_distance_per_frame
                * gaps[:, None]
            )

            geometry_cost = np.divide(
                distances,
                distance_thresholds,
                out=np.full_like(
                    distances,
                    INF_COST,
                ),
                where=distance_thresholds > 0,
            )

            fused_cost = (
                args.alpha
                * appearance_cost
                + (1.0 - args.alpha)
                * geometry_cost
            )

            fused_cost[
                distances > distance_thresholds
            ] = INF_COST

            accepted = solve_with_unmatched(
                real_cost=fused_cost,
                acceptance_cost=args.acceptance_cost,
            )
        else:
            similarities = np.empty(
                (
                    len(active_items),
                    len(detection_indices),
                ),
                dtype=np.float64,
            )

            distances = np.empty_like(
                similarities
            )

            gaps = np.empty(
                (len(active_items),),
                dtype=np.float64,
            )

            accepted = []

        matched_detection_positions: set[int] = set()

        for (
            track_position,
            detection_position,
            cost,
        ) in accepted:
            track_id, track = active_items[
                track_position
            ]

            global_detection_index = int(
                detection_indices[
                    detection_position
                ]
            )

            detection_embedding = (
                detection_embeddings[
                    detection_position
                ]
            )

            updated_embedding = (
                args.embedding_momentum
                * track.embedding
                + (
                    1.0
                    - args.embedding_momentum
                )
                * detection_embedding
            )

            updated_embedding = normalize_vector(
                updated_embedding
            )

            point = detection_xy[
                detection_position
            ]

            gap = int(
                frame_index
                - track.last_frame_index
            )

            active_tracks[track_id] = ActiveTrack(
                track_id=track_id,
                last_frame_index=int(
                    frame_index
                ),
                last_x_m=float(point[0]),
                last_y_m=float(point[1]),
                embedding=updated_embedding,
                observations=(
                    track.observations + 1
                ),
            )

            assigned_track_ids[
                global_detection_index
            ] = track_id

            assignment_costs[
                global_detection_index
            ] = cost

            assignment_similarities[
                global_detection_index
            ] = float(
                similarities[
                    track_position,
                    detection_position,
                ]
            )

            assignment_distances[
                global_detection_index
            ] = float(
                distances[
                    track_position,
                    detection_position,
                ]
            )

            assignment_gaps[
                global_detection_index
            ] = gap

            matched_detection_positions.add(
                detection_position
            )

        for detection_position, global_detection_index in enumerate(
            detection_indices
        ):
            if detection_position in matched_detection_positions:
                continue

            track_id = next_track_id
            next_track_id += 1

            point = detection_xy[
                detection_position
            ]

            active_tracks[track_id] = ActiveTrack(
                track_id=track_id,
                last_frame_index=int(
                    frame_index
                ),
                last_x_m=float(point[0]),
                last_y_m=float(point[1]),
                embedding=detection_embeddings[
                    detection_position
                ].copy(),
                observations=1,
            )

            assigned_track_ids[
                global_detection_index
            ] = track_id

            assignment_gaps[
                global_detection_index
            ] = 0

        print(
            f"Tracking: {frame_number}/{len(frame_indices)}",
            end="\r",
        )

    print()

    observations["track_id"] = assigned_track_ids
    observations["assignment_cost"] = assignment_costs
    observations["assignment_similarity"] = (
        assignment_similarities
    )
    observations["assignment_distance_m"] = (
        assignment_distances
    )
    observations["assignment_gap"] = assignment_gaps

    if (observations["track_id"] < 0).any():
        raise RuntimeError(
            "Часть детекций не получила track_id."
        )

    tracks_summary = build_tracks_summary(
        observations
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    observations.to_csv(
        args.output / "observations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    tracks_summary.to_csv(
        args.output / "tracks_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.save(
        args.output / "embeddings.npy",
        embeddings,
    )

    config = {
        "source_observations": str(
            observations_path
        ),
        "camera": args.camera,
        "detections": int(
            len(observations)
        ),
        "model_name": args.model_name,
        "weights": str(weights_path),
        "device": device,
        "alpha": args.alpha,
        "geometry_weight": (
            1.0 - args.alpha
        ),
        "max_distance_per_frame_m": (
            args.max_distance_per_frame
        ),
        "max_gap": args.max_gap,
        "acceptance_cost": (
            args.acceptance_cost
        ),
        "embedding_momentum": (
            args.embedding_momentum
        ),
    }

    with (
        args.output / "config.json"
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

    print(f"Custom tracks: {len(tracks_summary)}")
    print(
        "Median observations per track:",
        float(
            tracks_summary[
                "observations"
            ].median()
        ),
    )
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
        "Results:",
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
