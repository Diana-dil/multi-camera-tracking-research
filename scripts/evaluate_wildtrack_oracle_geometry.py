from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Сравнение appearance-only и oracle geometry "
            "на WILDTRACK."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/processed/wildtrack_C1_C6_50"
        ),
    )

    parser.add_argument(
        "--reid-results",
        type=Path,
        default=Path(
            "results/wildtrack_reid_C1_C6"
        ),
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(
            "data/raw/WILDTRACK/annotations_positions"
        ),
    )

    parser.add_argument(
        "--camera-a",
        default="C1",
    )

    parser.add_argument(
        "--camera-b",
        default="C6",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/wildtrack_oracle_geometry_C1_C6"
        ),
    )

    return parser.parse_args()


def normalize_frame_number(value: object) -> int:
    """
    В CSV кадр может быть записан как 0,
    а исходный файл WILDTRACK называется 00000000.json.
    """
    if isinstance(value, str):
        return int(value)

    return int(value)


def load_position_map(
    annotation_root: Path,
    frame_numbers: list[int],
) -> dict[tuple[int, int], int]:
    """
    Возвращает:
        (frame, person_id) -> position_id
    """
    position_map: dict[tuple[int, int], int] = {}

    for frame_number in frame_numbers:
        annotation_path = (
            annotation_root
            / f"{frame_number:08d}.json"
        )

        if not annotation_path.exists():
            raise FileNotFoundError(
                "Не найдена аннотация: "
                f"{annotation_path.resolve()}"
            )

        with annotation_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            annotations = json.load(file)

        if not isinstance(annotations, list):
            raise ValueError(
                f"Ожидался список объектов: {annotation_path}"
            )

        for person in annotations:
            if "personID" not in person:
                raise KeyError(
                    f"Нет personID в {annotation_path}"
                )

            if "positionID" not in person:
                raise KeyError(
                    "В аннотации отсутствует positionID. "
                    f"Доступные поля: {sorted(person.keys())}"
                )

            person_id = int(person["personID"])
            position_id = int(person["positionID"])

            position_map[
                (frame_number, person_id)
            ] = position_id

    return position_map


def add_positions(
    metadata: pd.DataFrame,
    position_map: dict[tuple[int, int], int],
) -> pd.DataFrame:
    result = metadata.copy()

    result["frame"] = result["frame"].apply(
        normalize_frame_number
    )

    result["person_id"] = (
        result["person_id"].astype(int)
    )

    result["position_id"] = [
        position_map.get(
            (
                int(row.frame),
                int(row.person_id),
            )
        )
        for row in result.itertuples()
    ]

    missing = result["position_id"].isna()

    if missing.any():
        examples = result.loc[
            missing,
            ["frame", "person_id", "camera_id"],
        ].head(10)

        raise ValueError(
            "Для части изображений не найден positionID:\n"
            f"{examples.to_string(index=False)}"
        )

    result["position_id"] = (
        result["position_id"].astype(int)
    )

    return result


def evaluate_direction(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    query_camera: str,
    gallery_camera: str,
) -> pd.DataFrame:
    rows: list[dict] = []

    for frame, frame_group in metadata.groupby(
        "frame",
        sort=True,
    ):
        query_group = frame_group[
            frame_group["camera_id"] == query_camera
        ]

        gallery_group = frame_group[
            frame_group["camera_id"] == gallery_camera
        ]

        if query_group.empty or gallery_group.empty:
            continue

        query_indices = (
            query_group["embedding_index"]
            .astype(int)
            .to_numpy()
        )

        gallery_indices = (
            gallery_group["embedding_index"]
            .astype(int)
            .to_numpy()
        )

        query_embeddings = embeddings[query_indices]
        gallery_embeddings = embeddings[gallery_indices]

        similarity_matrix = (
            query_embeddings
            @ gallery_embeddings.T
        )

        gallery_person_ids = (
            gallery_group["person_id"]
            .astype(int)
            .to_numpy()
        )

        gallery_position_ids = (
            gallery_group["position_id"]
            .astype(int)
            .to_numpy()
        )

        gallery_crop_paths = (
            gallery_group["crop_path"]
            .astype(str)
            .to_numpy()
        )

        for query_position, (
            _,
            query_row,
        ) in enumerate(query_group.iterrows()):
            query_person_id = int(
                query_row["person_id"]
            )

            query_position_id = int(
                query_row["position_id"]
            )

            # Оцениваем только человека, который присутствует
            # одновременно в обеих камерах.
            correct_gallery_positions = np.where(
                gallery_person_ids == query_person_id
            )[0]

            if len(correct_gallery_positions) == 0:
                continue

            similarities = similarity_matrix[
                query_position
            ]

            # Метод 1: только OSNet.
            appearance_index = int(
                np.argmax(similarities)
            )

            appearance_prediction = int(
                gallery_person_ids[
                    appearance_index
                ]
            )

            # Метод 2: oracle spatial gate.
            # Оставляем только людей с тем же positionID.
            geometry_candidates = np.where(
                gallery_position_ids
                == query_position_id
            )[0]

            geometry_fallback = False

            if len(geometry_candidates) == 0:
                # Теоретически для общего global ID
                # такого происходить не должно.
                geometry_fallback = True
                geometry_index = appearance_index
            else:
                best_local_position = int(
                    np.argmax(
                        similarities[
                            geometry_candidates
                        ]
                    )
                )

                geometry_index = int(
                    geometry_candidates[
                        best_local_position
                    ]
                )

            geometry_prediction = int(
                gallery_person_ids[
                    geometry_index
                ]
            )

            rows.append(
                {
                    "frame": int(frame),
                    "query_camera": query_camera,
                    "gallery_camera": gallery_camera,
                    "query_person_id": query_person_id,
                    "query_position_id": query_position_id,
                    "gallery_size": int(
                        len(gallery_group)
                    ),
                    "geometry_candidates": int(
                        len(geometry_candidates)
                    ),
                    "appearance_prediction": (
                        appearance_prediction
                    ),
                    "appearance_correct": int(
                        appearance_prediction
                        == query_person_id
                    ),
                    "appearance_similarity": float(
                        similarities[
                            appearance_index
                        ]
                    ),
                    "oracle_prediction": (
                        geometry_prediction
                    ),
                    "oracle_correct": int(
                        geometry_prediction
                        == query_person_id
                    ),
                    "oracle_similarity": float(
                        similarities[
                            geometry_index
                        ]
                    ),
                    "geometry_fallback": int(
                        geometry_fallback
                    ),
                    "query_crop": str(
                        query_row["crop_path"]
                    ),
                    "appearance_crop": str(
                        gallery_crop_paths[
                            appearance_index
                        ]
                    ),
                    "oracle_crop": str(
                        gallery_crop_paths[
                            geometry_index
                        ]
                    ),
                }
            )

    return pd.DataFrame(rows)


def summarize(
    results: pd.DataFrame,
    query_camera: str,
    gallery_camera: str,
) -> dict:
    if results.empty:
        raise RuntimeError(
            f"Нет результатов для "
            f"{query_camera} -> {gallery_camera}"
        )

    appearance_rank1 = float(
        results["appearance_correct"].mean()
    )

    oracle_rank1 = float(
        results["oracle_correct"].mean()
    )

    return {
        "query_camera": query_camera,
        "gallery_camera": gallery_camera,
        "queries": int(len(results)),
        "appearance_rank1": appearance_rank1,
        "oracle_geometry_rank1": oracle_rank1,
        "absolute_gain": (
            oracle_rank1 - appearance_rank1
        ),
        "relative_error_reduction": (
            (
                (1.0 - appearance_rank1)
                - (1.0 - oracle_rank1)
            )
            / (1.0 - appearance_rank1)
            if appearance_rank1 < 1.0
            else 0.0
        ),
        "mean_gallery_size": float(
            results["gallery_size"].mean()
        ),
        "mean_geometry_candidates": float(
            results[
                "geometry_candidates"
            ].mean()
        ),
        "geometry_fallbacks": int(
            results["geometry_fallback"].sum()
        ),
    }


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset.resolve()
    reid_root = args.reid_results.resolve()
    annotation_root = args.annotations.resolve()

    embedding_metadata_path = (
        reid_root / "embedding_metadata.csv"
    )

    embeddings_path = (
        reid_root / "embeddings.npy"
    )

    if not embedding_metadata_path.exists():
        raise FileNotFoundError(
            "Не найден embedding_metadata.csv: "
            f"{embedding_metadata_path}"
        )

    if not embeddings_path.exists():
        raise FileNotFoundError(
            "Не найден embeddings.npy: "
            f"{embeddings_path}"
        )

    metadata = pd.read_csv(
        embedding_metadata_path
    )

    embeddings = np.load(
        embeddings_path
    )

    if len(metadata) != len(embeddings):
        raise ValueError(
            "Количество строк metadata не совпадает "
            "с количеством embeddings: "
            f"{len(metadata)} != {len(embeddings)}"
        )

    if "embedding_index" not in metadata.columns:
        metadata["embedding_index"] = np.arange(
            len(metadata)
        )

    frame_numbers = sorted(
        {
            normalize_frame_number(value)
            for value in metadata["frame"]
        }
    )

    position_map = load_position_map(
        annotation_root=annotation_root,
        frame_numbers=frame_numbers,
    )

    metadata = add_positions(
        metadata=metadata,
        position_map=position_map,
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata.to_csv(
        args.output / "metadata_with_positions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    directions = [
        (args.camera_a, args.camera_b),
        (args.camera_b, args.camera_a),
    ]

    result_tables: list[pd.DataFrame] = []
    summaries: list[dict] = []

    for query_camera, gallery_camera in directions:
        direction_results = evaluate_direction(
            metadata=metadata,
            embeddings=embeddings,
            query_camera=query_camera,
            gallery_camera=gallery_camera,
        )

        direction_summary = summarize(
            results=direction_results,
            query_camera=query_camera,
            gallery_camera=gallery_camera,
        )

        result_tables.append(direction_results)
        summaries.append(direction_summary)

        print()
        print(
            f"{query_camera} -> {gallery_camera}"
        )
        print(
            "  Queries:",
            direction_summary["queries"],
        )
        print(
            "  Appearance Rank-1:",
            f"{direction_summary['appearance_rank1']:.4f}",
        )
        print(
            "  Oracle geometry Rank-1:",
            f"{direction_summary['oracle_geometry_rank1']:.4f}",
        )
        print(
            "  Absolute gain:",
            f"{direction_summary['absolute_gain']:.4f}",
        )
        print(
            "  Mean gallery size:",
            f"{direction_summary['mean_gallery_size']:.2f}",
        )
        print(
            "  Mean geometry candidates:",
            f"{direction_summary['mean_geometry_candidates']:.2f}",
        )
        print(
            "  Geometry fallbacks:",
            direction_summary["geometry_fallbacks"],
        )

    all_results = pd.concat(
        result_tables,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(
        summaries
    )

    all_results.to_csv(
        args.output / "per_query_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        args.output / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mean_appearance = float(
        summary_df["appearance_rank1"].mean()
    )

    mean_oracle = float(
        summary_df["oracle_geometry_rank1"].mean()
    )

    overall = {
        "dataset": str(dataset_root),
        "annotations": str(annotation_root),
        "queries": int(len(all_results)),
        "mean_appearance_rank1": mean_appearance,
        "mean_oracle_geometry_rank1": mean_oracle,
        "absolute_gain": (
            mean_oracle - mean_appearance
        ),
        "note": (
            "Oracle geometry uses ground-truth positionID "
            "and is an upper-bound experiment, not a "
            "deployable method."
        ),
        "directions": summaries,
    }

    with (
        args.output / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            overall,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("Итог:")
    print(
        "  Mean appearance Rank-1:",
        f"{mean_appearance:.4f}",
    )
    print(
        "  Mean oracle geometry Rank-1:",
        f"{mean_oracle:.4f}",
    )
    print(
        "  Gain:",
        f"{mean_oracle - mean_appearance:.4f}",
    )
    print()
    print(
        "Результаты:",
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()