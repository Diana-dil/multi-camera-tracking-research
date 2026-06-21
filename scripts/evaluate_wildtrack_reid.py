from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TORCHREID_ROOT = PROJECT_ROOT / "external" / "deep-person-reid"

if str(TORCHREID_ROOT) not in sys.path:
    sys.path.insert(0, str(TORCHREID_ROOT))

from torchreid.utils import FeatureExtractor  # noqa: E402


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
            "Оценка межкамерной идентификации людей "
            "на подготовленном наборе WILDTRACK."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/wildtrack_C1_C6_50"),
        help="Папка с metadata.csv и crops.",
    )

    parser.add_argument(
        "--camera-a",
        default="C1",
        help="Первая камера.",
    )

    parser.add_argument(
        "--camera-b",
        default="C6",
        help="Вторая камера.",
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Путь к весам OSNet.",
    )

    parser.add_argument(
        "--model-name",
        default="osnet_ain_x1_0",
        help="Название модели Torchreid.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Размер пакета при извлечении признаков.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/wildtrack_reid_C1_C6"),
        help="Папка сохранения результатов.",
    )

    return parser.parse_args()


def resolve_crop_path(
    dataset_root: Path,
    relative_path: str,
) -> Path:
    path = dataset_root / Path(relative_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Изображение не найдено: {path.resolve()}"
        )

    return path.resolve()


def extract_embeddings(
    metadata: pd.DataFrame,
    dataset_root: Path,
    extractor: FeatureExtractor,
    batch_size: int,
) -> np.ndarray:
    image_paths = [
        str(resolve_crop_path(dataset_root, value))
        for value in metadata["crop_path"].tolist()
    ]

    all_embeddings: list[torch.Tensor] = []

    total = len(image_paths)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_paths = image_paths[start:end]

        with torch.no_grad():
            batch_embeddings = extractor(batch_paths)

        batch_embeddings = F.normalize(
            batch_embeddings.float(),
            p=2,
            dim=1,
        )

        all_embeddings.append(
            batch_embeddings.detach().cpu()
        )

        print(
            f"Embeddings: {end}/{total}",
            end="\r",
        )

    print()

    embeddings = torch.cat(
        all_embeddings,
        dim=0,
    ).numpy()

    return embeddings


def evaluate_direction(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    query_camera: str,
    gallery_camera: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    retrieval_rows: list[dict] = []
    pair_rows: list[dict] = []

    grouped = metadata.groupby("frame", sort=True)

    for frame, frame_group in grouped:
        query_group = frame_group[
            frame_group["camera_id"] == query_camera
        ]

        gallery_group = frame_group[
            frame_group["camera_id"] == gallery_camera
        ]

        if query_group.empty or gallery_group.empty:
            continue

        query_indices = query_group.index.to_numpy()
        gallery_indices = gallery_group.index.to_numpy()

        query_embeddings = embeddings[query_indices]
        gallery_embeddings = embeddings[gallery_indices]

        similarity_matrix = (
            query_embeddings @ gallery_embeddings.T
        )

        gallery_ids = gallery_group[
            "person_id"
        ].astype(int).to_numpy()

        for query_position, (
            query_index,
            query_row,
        ) in enumerate(query_group.iterrows()):
            query_person_id = int(
                query_row["person_id"]
            )

            positive_positions = np.where(
                gallery_ids == query_person_id
            )[0]

            # Оцениваем только человека, который действительно
            # присутствует одновременно в обеих камерах.
            if len(positive_positions) == 0:
                continue

            similarities = similarity_matrix[
                query_position
            ]

            ranking = np.argsort(
                -similarities
            )

            ranked_gallery_ids = gallery_ids[ranking]

            positive_rank_positions = np.where(
                ranked_gallery_ids == query_person_id
            )[0]

            rank = int(
                positive_rank_positions[0] + 1
            )

            top1_position = int(ranking[0])
            predicted_person_id = int(
                gallery_ids[top1_position]
            )

            positive_similarity = float(
                similarities[
                    positive_positions[0]
                ]
            )

            top1_similarity = float(
                similarities[top1_position]
            )

            retrieval_rows.append(
                {
                    "frame": frame,
                    "query_camera": query_camera,
                    "gallery_camera": gallery_camera,
                    "query_person_id": query_person_id,
                    "predicted_person_id": (
                        predicted_person_id
                    ),
                    "rank": rank,
                    "rank1_correct": int(rank == 1),
                    "rank5_correct": int(rank <= 5),
                    "reciprocal_rank": 1.0 / rank,
                    "positive_similarity": (
                        positive_similarity
                    ),
                    "top1_similarity": top1_similarity,
                    "query_crop": query_row[
                        "crop_path"
                    ],
                    "predicted_crop": (
                        gallery_group.iloc[
                            top1_position
                        ]["crop_path"]
                    ),
                }
            )

            for gallery_position, (
                gallery_index,
                gallery_row,
            ) in enumerate(
                gallery_group.iterrows()
            ):
                gallery_person_id = int(
                    gallery_row["person_id"]
                )

                pair_rows.append(
                    {
                        "frame": frame,
                        "query_camera": query_camera,
                        "gallery_camera": gallery_camera,
                        "query_person_id": (
                            query_person_id
                        ),
                        "gallery_person_id": (
                            gallery_person_id
                        ),
                        "similarity": float(
                            similarities[
                                gallery_position
                            ]
                        ),
                        "label": int(
                            query_person_id
                            == gallery_person_id
                        ),
                        "query_crop": query_row[
                            "crop_path"
                        ],
                        "gallery_crop": gallery_row[
                            "crop_path"
                        ],
                    }
                )

    return (
        pd.DataFrame(retrieval_rows),
        pd.DataFrame(pair_rows),
    )


def find_best_threshold(
    pairs: pd.DataFrame,
) -> dict:
    if pairs.empty:
        return {
            "threshold": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }

    similarities = pairs[
        "similarity"
    ].to_numpy()

    labels = pairs[
        "label"
    ].astype(int).to_numpy()

    thresholds = np.linspace(
        float(similarities.min()),
        float(similarities.max()),
        501,
    )

    best_result = {
        "threshold": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    for threshold in thresholds:
        predictions = (
            similarities >= threshold
        ).astype(int)

        true_positive = int(
            ((predictions == 1) & (labels == 1)).sum()
        )

        false_positive = int(
            ((predictions == 1) & (labels == 0)).sum()
        )

        false_negative = int(
            ((predictions == 0) & (labels == 1)).sum()
        )

        precision = (
            true_positive
            / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )

        recall = (
            true_positive
            / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        if f1 > best_result["f1"]:
            best_result = {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }

    return best_result


def summarize_direction(
    retrieval: pd.DataFrame,
    pairs: pd.DataFrame,
    query_camera: str,
    gallery_camera: str,
) -> dict:
    positive_pairs = pairs[
        pairs["label"] == 1
    ]

    negative_pairs = pairs[
        pairs["label"] == 0
    ]

    threshold_metrics = find_best_threshold(
        pairs
    )

    return {
        "query_camera": query_camera,
        "gallery_camera": gallery_camera,
        "queries": int(len(retrieval)),
        "rank1": float(
            retrieval["rank1_correct"].mean()
        ),
        "rank5": float(
            retrieval["rank5_correct"].mean()
        ),
        "mrr": float(
            retrieval["reciprocal_rank"].mean()
        ),
        "mean_positive_similarity": float(
            positive_pairs[
                "similarity"
            ].mean()
        ),
        "mean_negative_similarity": float(
            negative_pairs[
                "similarity"
            ].mean()
        ),
        "best_threshold": (
            threshold_metrics["threshold"]
        ),
        "threshold_precision": (
            threshold_metrics["precision"]
        ),
        "threshold_recall": (
            threshold_metrics["recall"]
        ),
        "threshold_f1": (
            threshold_metrics["f1"]
        ),
    }


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset.resolve()
    metadata_path = dataset_root / "metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.csv не найден: "
            f"{metadata_path}"
        )

    weights_path = args.weights.resolve()

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Веса модели не найдены: "
            f"{weights_path}"
        )

    metadata = pd.read_csv(
        metadata_path
    ).reset_index(drop=True)

    required_columns = {
        "frame",
        "camera_id",
        "person_id",
        "crop_path",
    }

    missing_columns = (
        required_columns
        - set(metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            f"В metadata.csv отсутствуют поля: "
            f"{sorted(missing_columns)}"
        )

    available_cameras = set(
        metadata["camera_id"].unique()
    )

    for camera in (
        args.camera_a,
        args.camera_b,
    ):
        if camera not in available_cameras:
            raise ValueError(
                f"Камера {camera} отсутствует "
                f"в metadata.csv"
            )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Weights: {weights_path}")
    print(f"Images: {len(metadata)}")
    print()

    extractor = FeatureExtractor(
        model_name=args.model_name,
        model_path=str(weights_path),
        device=device,
    )

    embeddings = extract_embeddings(
        metadata=metadata,
        dataset_root=dataset_root,
        extractor=extractor,
        batch_size=args.batch_size,
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        args.output / "embeddings.npy",
        embeddings,
    )

    metadata_with_index = metadata.copy()
    metadata_with_index[
        "embedding_index"
    ] = np.arange(len(metadata))

    metadata_with_index.to_csv(
        args.output / "embedding_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_retrieval: list[pd.DataFrame] = []
    all_pairs: list[pd.DataFrame] = []
    summaries: list[dict] = []

    directions = [
        (args.camera_a, args.camera_b),
        (args.camera_b, args.camera_a),
    ]

    for query_camera, gallery_camera in directions:
        retrieval, pairs = evaluate_direction(
            metadata=metadata,
            embeddings=embeddings,
            query_camera=query_camera,
            gallery_camera=gallery_camera,
        )

        summary = summarize_direction(
            retrieval=retrieval,
            pairs=pairs,
            query_camera=query_camera,
            gallery_camera=gallery_camera,
        )

        all_retrieval.append(retrieval)
        all_pairs.append(pairs)
        summaries.append(summary)

        print()
        print(
            f"{query_camera} -> "
            f"{gallery_camera}"
        )
        print(
            f"  Queries: {summary['queries']}"
        )
        print(
            f"  Rank-1: {summary['rank1']:.4f}"
        )
        print(
            f"  Rank-5: {summary['rank5']:.4f}"
        )
        print(
            f"  MRR: {summary['mrr']:.4f}"
        )
        print(
            "  Mean positive similarity: "
            f"{summary['mean_positive_similarity']:.4f}"
        )
        print(
            "  Mean negative similarity: "
            f"{summary['mean_negative_similarity']:.4f}"
        )
        print(
            "  Best threshold: "
            f"{summary['best_threshold']:.4f}"
        )
        print(
            "  Threshold F1: "
            f"{summary['threshold_f1']:.4f}"
        )

    retrieval_df = pd.concat(
        all_retrieval,
        ignore_index=True,
    )

    pairs_df = pd.concat(
        all_pairs,
        ignore_index=True,
    )

    retrieval_df.to_csv(
        args.output / "retrieval_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pairs_df.to_csv(
        args.output / "pair_similarities.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df.to_csv(
        args.output / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall_summary = {
        "dataset": str(dataset_root),
        "model_name": args.model_name,
        "weights": str(weights_path),
        "device": device,
        "images": int(len(metadata)),
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
        "directions": summaries,
        "mean_rank1": float(
            summary_df["rank1"].mean()
        ),
        "mean_rank5": float(
            summary_df["rank5"].mean()
        ),
        "mean_mrr": float(
            summary_df["mrr"].mean()
        ),
    }

    with (
        args.output / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            overall_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "Средний Rank-1:",
        f"{overall_summary['mean_rank1']:.4f}",
    )
    print(
        "Средний Rank-5:",
        f"{overall_summary['mean_rank5']:.4f}",
    )
    print(
        "Средний MRR:",
        f"{overall_summary['mean_mrr']:.4f}",
    )
    print()
    print(
        "Результаты сохранены:",
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()