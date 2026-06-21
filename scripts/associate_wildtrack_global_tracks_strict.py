from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


INF_COST = 1_000_000.0


@dataclass
class EdgeCandidate:
    camera_a: str
    track_a: int
    camera_b: str
    track_b: int
    votes: int
    overlap_frames: int
    support_ratio: float
    mean_similarity: float
    median_distance_m: float
    mean_cost: float


class UnionFind:
    def __init__(
        self,
        nodes: list[tuple[str, int]],
        frame_sets: dict[tuple[str, int], set[int]],
    ) -> None:
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}

        self.component_frames: dict[
            tuple[str, int],
            dict[str, set[int]],
        ] = {}

        for node in nodes:
            camera, _ = node
            self.component_frames[node] = {
                camera: set(frame_sets[node])
            }

    def find(
        self,
        node: tuple[str, int],
    ) -> tuple[str, int]:
        parent = self.parent[node]

        if parent != node:
            self.parent[node] = self.find(parent)

        return self.parent[node]

    def can_union(
        self,
        node_a: tuple[str, int],
        node_b: tuple[str, int],
        max_same_camera_overlap: int,
    ) -> tuple[bool, str]:
        root_a = self.find(node_a)
        root_b = self.find(node_b)

        if root_a == root_b:
            return True, "already_connected"

        frames_a = self.component_frames[root_a]
        frames_b = self.component_frames[root_b]

        common_cameras = (
            set(frames_a)
            & set(frames_b)
        )

        for camera in common_cameras:
            overlap = len(
                frames_a[camera]
                & frames_b[camera]
            )

            if overlap > max_same_camera_overlap:
                return (
                    False,
                    (
                        "same_camera_temporal_conflict:"
                        f"{camera}:{overlap}"
                    ),
                )

        return True, "ok"

    def union(
        self,
        node_a: tuple[str, int],
        node_b: tuple[str, int],
    ) -> None:
        root_a = self.find(node_a)
        root_b = self.find(node_b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a

        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1

        merged_frames: dict[str, set[int]] = {}

        all_cameras = (
            set(self.component_frames[root_a])
            | set(self.component_frames[root_b])
        )

        for camera in all_cameras:
            merged_frames[camera] = (
                set(
                    self.component_frames[
                        root_a
                    ].get(camera, set())
                )
                | set(
                    self.component_frames[
                        root_b
                    ].get(camera, set())
                )
            )

        self.component_frames[root_a] = (
            merged_frames
        )

        del self.component_frames[root_b]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Объединение локальных треков двух камер "
            "WILDTRACK в глобальные идентификаторы."
        )
    )

    parser.add_argument(
        "--camera-a-dir",
        type=Path,
        required=True,
        help=(
            "Папка результата собственного трекера "
            "для первой камеры."
        ),
    )

    parser.add_argument(
        "--camera-b-dir",
        type=Path,
        required=True,
        help=(
            "Папка результата собственного трекера "
            "для второй камеры."
        ),
    )

    parser.add_argument(
        "--camera-a",
        default="C1",
    )

    parser.add_argument(
        "--camera-b",
        default="C3",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.75,
        help="Вес appearance-cost.",
    )

    parser.add_argument(
        "--geometry-threshold",
        type=float,
        default=2.0,
        help=(
            "Максимальное допустимое расстояние "
            "между детекциями на одном кадре, м."
        ),
    )

    parser.add_argument(
        "--acceptance-cost",
        type=float,
        default=0.50,
        help=(
            "Максимальная стоимость принятой "
            "покадровой пары."
        ),
    )

    parser.add_argument(
        "--min-votes",
        type=int,
        default=2,
        help=(
            "Минимальное число покадровых подтверждений "
            "между двумя локальными треками."
        ),
    )

    parser.add_argument(
        "--min-support-ratio",
        type=float,
        default=0.30,
        help=(
            "Минимальная доля совпавших кадров "
            "от числа кадров пересечения tracklet."
        ),
    )

    parser.add_argument(
        "--min-mean-similarity",
        type=float,
        default=0.55,
        help=(
            "Минимальное среднее косинусное сходство "
            "между двумя tracklet."
        ),
    )

    parser.add_argument(
        "--max-median-distance",
        type=float,
        default=0.75,
        help=(
            "Максимальное медианное расстояние между "
            "связанными наблюдениями tracklet, м."
        ),
    )

    parser.add_argument(
        "--max-same-camera-overlap",
        type=int,
        default=0,
        help=(
            "Допустимое число общих кадров у двух "
            "tracklet одной камеры внутри global ID."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def normalize_embeddings(
    embeddings: np.ndarray,
) -> np.ndarray:
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    return np.divide(
        embeddings,
        norms,
        out=np.zeros_like(
            embeddings,
            dtype=np.float32,
        ),
        where=norms > 1e-12,
    )


def load_camera_data(
    result_dir: Path,
    expected_camera: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    observations_path = (
        result_dir / "observations.csv"
    )

    embeddings_path = (
        result_dir / "embeddings.npy"
    )

    if not observations_path.exists():
        raise FileNotFoundError(
            observations_path.resolve()
        )

    if not embeddings_path.exists():
        raise FileNotFoundError(
            embeddings_path.resolve()
        )

    observations = (
        pd.read_csv(observations_path)
        .reset_index(drop=True)
    )

    embeddings = np.load(
        embeddings_path
    )

    required_columns = {
        "camera_id",
        "frame_index",
        "frame_name",
        "track_id",
        "confidence",
        "ground_x_m",
        "ground_y_m",
    }

    missing = (
        required_columns
        - set(observations.columns)
    )

    if missing:
        raise ValueError(
            f"В {observations_path} отсутствуют поля: "
            f"{sorted(missing)}"
        )

    observations = observations[
        observations["camera_id"]
        == expected_camera
    ].copy()

    if observations.empty:
        raise ValueError(
            f"В {observations_path} нет камеры "
            f"{expected_camera}."
        )

    observations["frame_index"] = (
        observations["frame_index"].astype(int)
    )

    observations["track_id"] = (
        observations["track_id"].astype(int)
    )

    if "embedding_index" in observations.columns:
        embedding_indices = (
            observations["embedding_index"]
            .astype(int)
            .to_numpy()
        )
    else:
        if len(observations) != len(embeddings):
            raise ValueError(
                "Нет embedding_index, а число строк "
                "observations не совпадает с embeddings."
            )

        embedding_indices = np.arange(
            len(observations)
        )

    if embedding_indices.max() >= len(embeddings):
        raise ValueError(
            "embedding_index выходит за пределы "
            "embeddings.npy."
        )

    selected_embeddings = embeddings[
        embedding_indices
    ]

    selected_embeddings = normalize_embeddings(
        selected_embeddings.astype(
            np.float32
        )
    )

    observations = observations.reset_index(
        drop=True
    )

    observations[
        "local_embedding_index"
    ] = np.arange(len(observations))

    return observations, selected_embeddings


def solve_with_unmatched(
    real_cost: np.ndarray,
    acceptance_cost: float,
) -> list[tuple[int, int, float]]:
    count_a, count_b = real_cost.shape

    if count_a == 0 or count_b == 0:
        return []

    total_size = count_a + count_b

    augmented = np.full(
        (total_size, total_size),
        INF_COST,
        dtype=np.float64,
    )

    augmented[
        :count_a,
        :count_b,
    ] = real_cost

    dummy_cost = acceptance_cost / 2.0

    for index_a in range(count_a):
        augmented[
            index_a,
            count_b + index_a,
        ] = dummy_cost

    for index_b in range(count_b):
        augmented[
            count_a + index_b,
            index_b,
        ] = dummy_cost

    augmented[
        count_a:,
        count_b:,
    ] = 0.0

    row_indices, column_indices = (
        linear_sum_assignment(augmented)
    )

    accepted: list[
        tuple[int, int, float]
    ] = []

    for row_index, column_index in zip(
        row_indices,
        column_indices,
    ):
        if (
            row_index < count_a
            and column_index < count_b
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


def match_frame(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    alpha: float,
    geometry_threshold: float,
    acceptance_cost: float,
) -> list[dict]:
    if frame_a.empty or frame_b.empty:
        return []

    embedding_indices_a = (
        frame_a["local_embedding_index"]
        .astype(int)
        .to_numpy()
    )

    embedding_indices_b = (
        frame_b["local_embedding_index"]
        .astype(int)
        .to_numpy()
    )

    feature_a = embeddings_a[
        embedding_indices_a
    ]

    feature_b = embeddings_b[
        embedding_indices_b
    ]

    similarities = feature_a @ feature_b.T
    appearance_cost = 1.0 - similarities

    points_a = frame_a[
        ["ground_x_m", "ground_y_m"]
    ].to_numpy(dtype=np.float64)

    points_b = frame_b[
        ["ground_x_m", "ground_y_m"]
    ].to_numpy(dtype=np.float64)

    distances = np.linalg.norm(
        points_a[:, None, :]
        - points_b[None, :, :],
        axis=2,
    )

    geometry_cost = (
        distances / geometry_threshold
    )

    fused_cost = (
        alpha * appearance_cost
        + (1.0 - alpha) * geometry_cost
    )

    fused_cost[
        distances > geometry_threshold
    ] = INF_COST

    assignments = solve_with_unmatched(
        real_cost=fused_cost,
        acceptance_cost=acceptance_cost,
    )

    rows: list[dict] = []

    for (
        position_a,
        position_b,
        cost,
    ) in assignments:
        row_a = frame_a.iloc[position_a]
        row_b = frame_b.iloc[position_b]

        rows.append(
            {
                "frame_index": int(
                    row_a["frame_index"]
                ),
                "track_a": int(
                    row_a["track_id"]
                ),
                "track_b": int(
                    row_b["track_id"]
                ),
                "similarity": float(
                    similarities[
                        position_a,
                        position_b,
                    ]
                ),
                "distance_m": float(
                    distances[
                        position_a,
                        position_b,
                    ]
                ),
                "cost": float(cost),
            }
        )

    return rows


def build_track_frame_sets(
    observations: pd.DataFrame,
) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}

    for track_id, group in observations.groupby(
        "track_id"
    ):
        result[int(track_id)] = set(
            group["frame_index"]
            .astype(int)
            .tolist()
        )

    return result


def aggregate_tracklet_edges(
    frame_matches: pd.DataFrame,
    frames_a: dict[int, set[int]],
    frames_b: dict[int, set[int]],
    camera_a: str,
    camera_b: str,
) -> list[EdgeCandidate]:
    candidates: list[EdgeCandidate] = []

    if frame_matches.empty:
        return candidates

    for (
        track_a,
        track_b,
    ), group in frame_matches.groupby(
        ["track_a", "track_b"],
        sort=True,
    ):
        track_a = int(track_a)
        track_b = int(track_b)

        overlap_frames = len(
            frames_a[track_a]
            & frames_b[track_b]
        )

        votes = int(len(group))

        support_ratio = (
            votes / overlap_frames
            if overlap_frames > 0
            else 0.0
        )

        candidates.append(
            EdgeCandidate(
                camera_a=camera_a,
                track_a=track_a,
                camera_b=camera_b,
                track_b=track_b,
                votes=votes,
                overlap_frames=(
                    overlap_frames
                ),
                support_ratio=float(
                    support_ratio
                ),
                mean_similarity=float(
                    group["similarity"].mean()
                ),
                median_distance_m=float(
                    group["distance_m"].median()
                ),
                mean_cost=float(
                    group["cost"].mean()
                ),
            )
        )

    return candidates


def create_global_mapping(
    observations_a: pd.DataFrame,
    observations_b: pd.DataFrame,
    candidates: list[EdgeCandidate],
    camera_a: str,
    camera_b: str,
    min_votes: int,
    min_support_ratio: float,
    min_mean_similarity: float,
    max_median_distance: float,
    max_same_camera_overlap: int,
) -> tuple[
    dict[tuple[str, int], int],
    pd.DataFrame,
]:
    frames_a = build_track_frame_sets(
        observations_a
    )

    frames_b = build_track_frame_sets(
        observations_b
    )

    frame_sets: dict[
        tuple[str, int],
        set[int],
    ] = {}

    for track_id, frames in frames_a.items():
        frame_sets[
            (camera_a, track_id)
        ] = frames

    for track_id, frames in frames_b.items():
        frame_sets[
            (camera_b, track_id)
        ] = frames

    nodes = sorted(frame_sets)

    union_find = UnionFind(
        nodes=nodes,
        frame_sets=frame_sets,
    )

    ordered_candidates = sorted(
        candidates,
        key=lambda edge: (
            -edge.votes,
            -edge.support_ratio,
            edge.mean_cost,
            -edge.mean_similarity,
            edge.median_distance_m,
        ),
    )

    edge_rows: list[dict] = []

    for edge in ordered_candidates:
        rejection_reasons: list[str] = []

        if edge.votes < min_votes:
            rejection_reasons.append(
                f"votes:{edge.votes}<{min_votes}"
            )

        if edge.support_ratio < min_support_ratio:
            rejection_reasons.append(
                "support:"
                f"{edge.support_ratio:.3f}"
                f"<{min_support_ratio:.3f}"
            )

        if edge.mean_similarity < min_mean_similarity:
            rejection_reasons.append(
                "similarity:"
                f"{edge.mean_similarity:.3f}"
                f"<{min_mean_similarity:.3f}"
            )

        if edge.median_distance_m > max_median_distance:
            rejection_reasons.append(
                "distance:"
                f"{edge.median_distance_m:.3f}"
                f">{max_median_distance:.3f}"
            )

        accepted_by_threshold = (
            len(rejection_reasons) == 0
        )

        node_a = (
            edge.camera_a,
            edge.track_a,
        )

        node_b = (
            edge.camera_b,
            edge.track_b,
        )

        if not accepted_by_threshold:
            accepted = False
            reason = ";".join(
                rejection_reasons
            )
        else:
            can_union, reason = (
                union_find.can_union(
                    node_a=node_a,
                    node_b=node_b,
                    max_same_camera_overlap=(
                        max_same_camera_overlap
                    ),
                )
            )

            accepted = can_union

            if accepted:
                union_find.union(
                    node_a,
                    node_b,
                )

        edge_rows.append(
            {
                "camera_a": edge.camera_a,
                "track_a": edge.track_a,
                "camera_b": edge.camera_b,
                "track_b": edge.track_b,
                "votes": edge.votes,
                "overlap_frames": (
                    edge.overlap_frames
                ),
                "support_ratio": (
                    edge.support_ratio
                ),
                "mean_similarity": (
                    edge.mean_similarity
                ),
                "median_distance_m": (
                    edge.median_distance_m
                ),
                "mean_cost": (
                    edge.mean_cost
                ),
                "accepted": int(accepted),
                "reason": reason,
            }
        )

    components: dict[
        tuple[str, int],
        list[tuple[str, int]],
    ] = {}

    for node in nodes:
        root = union_find.find(node)

        components.setdefault(
            root,
            [],
        ).append(node)

    def component_sort_key(
        component_nodes: list[
            tuple[str, int]
        ],
    ) -> tuple[int, str, int]:
        first_frame = min(
            min(frame_sets[node])
            for node in component_nodes
        )

        first_node = min(
            component_nodes
        )

        return (
            first_frame,
            first_node[0],
            first_node[1],
        )

    sorted_components = sorted(
        components.values(),
        key=component_sort_key,
    )

    mapping: dict[
        tuple[str, int],
        int,
    ] = {}

    for global_id, component_nodes in enumerate(
        sorted_components,
        start=1,
    ):
        for node in component_nodes:
            mapping[node] = global_id

    return mapping, pd.DataFrame(edge_rows)


def build_global_tracks(
    global_observations: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    component_sizes = (
        global_observations[
            [
                "global_id",
                "camera_id",
                "track_id",
            ]
        ]
        .drop_duplicates()
        .groupby("global_id")
        .size()
        .to_dict()
    )

    component_cameras = (
        global_observations[
            [
                "global_id",
                "camera_id",
            ]
        ]
        .drop_duplicates()
        .groupby("global_id")[
            "camera_id"
        ]
        .apply(
            lambda values: ",".join(
                sorted(values.astype(str))
            )
        )
        .to_dict()
    )

    for (
        global_id,
        camera_id,
        track_id,
    ), group in global_observations.groupby(
        [
            "global_id",
            "camera_id",
            "track_id",
        ],
        sort=True,
    ):
        ordered = group.sort_values(
            "frame_index"
        )

        rows.append(
            {
                "global_id": int(global_id),
                "camera_id": str(camera_id),
                "local_track_id": int(
                    track_id
                ),
                "first_frame_index": int(
                    ordered[
                        "frame_index"
                    ].min()
                ),
                "last_frame_index": int(
                    ordered[
                        "frame_index"
                    ].max()
                ),
                "observations": int(
                    len(ordered)
                ),
                "mean_confidence": float(
                    ordered[
                        "confidence"
                    ].mean()
                ),
                "mean_ground_x_m": float(
                    ordered[
                        "ground_x_m"
                    ].mean()
                ),
                "mean_ground_y_m": float(
                    ordered[
                        "ground_y_m"
                    ].mean()
                ),
                "component_tracklets": int(
                    component_sizes[
                        int(global_id)
                    ]
                ),
                "component_cameras": (
                    component_cameras[
                        int(global_id)
                    ]
                ),
                "is_cross_camera": int(
                    "," in component_cameras[
                        int(global_id)
                    ]
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(
            "--alpha должен быть в диапазоне [0, 1]."
        )

    if args.geometry_threshold <= 0:
        raise ValueError(
            "--geometry-threshold должен быть > 0."
        )

    if args.min_votes < 1:
        raise ValueError(
            "--min-votes должен быть >= 1."
        )

    if not 0.0 <= args.min_support_ratio <= 1.0:
        raise ValueError(
            "--min-support-ratio должен быть "
            "в диапазоне [0, 1]."
        )

    if not -1.0 <= args.min_mean_similarity <= 1.0:
        raise ValueError(
            "--min-mean-similarity должен быть "
            "в диапазоне [-1, 1]."
        )

    if args.max_median_distance <= 0:
        raise ValueError(
            "--max-median-distance должен быть > 0."
        )

    observations_a, embeddings_a = (
        load_camera_data(
            result_dir=(
                args.camera_a_dir.resolve()
            ),
            expected_camera=args.camera_a,
        )
    )

    observations_b, embeddings_b = (
        load_camera_data(
            result_dir=(
                args.camera_b_dir.resolve()
            ),
            expected_camera=args.camera_b,
        )
    )

    frames_a = set(
        observations_a[
            "frame_index"
        ].astype(int)
    )

    frames_b = set(
        observations_b[
            "frame_index"
        ].astype(int)
    )

    common_frames = sorted(
        frames_a & frames_b
    )

    if not common_frames:
        raise ValueError(
            "У камер нет общих frame_index."
        )

    print(
        f"{args.camera_a}: "
        f"{len(observations_a)} observations, "
        f"{observations_a['track_id'].nunique()} tracks"
    )

    print(
        f"{args.camera_b}: "
        f"{len(observations_b)} observations, "
        f"{observations_b['track_id'].nunique()} tracks"
    )

    print(
        f"Common frames: {len(common_frames)}"
    )

    frame_match_rows: list[dict] = []

    for number, frame_index in enumerate(
        common_frames,
        start=1,
    ):
        frame_a = (
            observations_a[
                observations_a[
                    "frame_index"
                ] == frame_index
            ]
            .reset_index(drop=True)
        )

        frame_b = (
            observations_b[
                observations_b[
                    "frame_index"
                ] == frame_index
            ]
            .reset_index(drop=True)
        )

        matches = match_frame(
            frame_a=frame_a,
            frame_b=frame_b,
            embeddings_a=embeddings_a,
            embeddings_b=embeddings_b,
            alpha=args.alpha,
            geometry_threshold=(
                args.geometry_threshold
            ),
            acceptance_cost=(
                args.acceptance_cost
            ),
        )

        frame_match_rows.extend(matches)

        print(
            f"Frame matching: "
            f"{number}/{len(common_frames)}",
            end="\r",
        )

    print()

    frame_matches = pd.DataFrame(
        frame_match_rows,
        columns=[
            "frame_index",
            "track_a",
            "track_b",
            "similarity",
            "distance_m",
            "cost",
        ],
    )

    track_frames_a = build_track_frame_sets(
        observations_a
    )

    track_frames_b = build_track_frame_sets(
        observations_b
    )

    candidates = aggregate_tracklet_edges(
        frame_matches=frame_matches,
        frames_a=track_frames_a,
        frames_b=track_frames_b,
        camera_a=args.camera_a,
        camera_b=args.camera_b,
    )

    mapping, tracklet_matches = (
        create_global_mapping(
            observations_a=observations_a,
            observations_b=observations_b,
            candidates=candidates,
            camera_a=args.camera_a,
            camera_b=args.camera_b,
            min_votes=args.min_votes,
            min_support_ratio=(
                args.min_support_ratio
            ),
            min_mean_similarity=(
                args.min_mean_similarity
            ),
            max_median_distance=(
                args.max_median_distance
            ),
            max_same_camera_overlap=(
                args.max_same_camera_overlap
            ),
        )
    )

    output_a = observations_a.copy()
    output_b = observations_b.copy()

    output_a["global_id"] = [
        mapping[
            (
                args.camera_a,
                int(track_id),
            )
        ]
        for track_id in output_a[
            "track_id"
        ]
    ]

    output_b["global_id"] = [
        mapping[
            (
                args.camera_b,
                int(track_id),
            )
        ]
        for track_id in output_b[
            "track_id"
        ]
    ]

    global_observations = pd.concat(
        [
            output_a,
            output_b,
        ],
        ignore_index=True,
    ).sort_values(
        [
            "frame_index",
            "camera_id",
            "global_id",
        ]
    )

    global_tracks = build_global_tracks(
        global_observations
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_matches.to_csv(
        args.output / "frame_matches.csv",
        index=False,
        encoding="utf-8-sig",
    )

    tracklet_matches.to_csv(
        args.output / "tracklet_matches.csv",
        index=False,
        encoding="utf-8-sig",
    )

    global_observations.to_csv(
        args.output / "global_observations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    global_tracks.to_csv(
        args.output / "global_tracks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    accepted_edges = (
        int(
            tracklet_matches[
                "accepted"
            ].sum()
        )
        if not tracklet_matches.empty
        else 0
    )

    global_id_count = int(
        global_tracks[
            "global_id"
        ].nunique()
    )

    cross_camera_ids = int(
        global_tracks.loc[
            global_tracks[
                "is_cross_camera"
            ] == 1,
            "global_id",
        ].nunique()
    )

    singleton_ids = int(
        global_tracks.loc[
            global_tracks[
                "component_tracklets"
            ] == 1,
            "global_id",
        ].nunique()
    )

    multi_tracklet_ids = int(
        global_tracks.loc[
            global_tracks[
                "component_tracklets"
            ] > 1,
            "global_id",
        ].nunique()
    )

    summary = {
        "camera_a": args.camera_a,
        "camera_b": args.camera_b,
        "camera_a_observations": int(
            len(observations_a)
        ),
        "camera_b_observations": int(
            len(observations_b)
        ),
        "camera_a_local_tracks": int(
            observations_a[
                "track_id"
            ].nunique()
        ),
        "camera_b_local_tracks": int(
            observations_b[
                "track_id"
            ].nunique()
        ),
        "common_frames": int(
            len(common_frames)
        ),
        "frame_level_matches": int(
            len(frame_matches)
        ),
        "candidate_tracklet_edges": int(
            len(tracklet_matches)
        ),
        "accepted_tracklet_edges": (
            accepted_edges
        ),
        "global_ids": global_id_count,
        "cross_camera_global_ids": (
            cross_camera_ids
        ),
        "singleton_global_ids": (
            singleton_ids
        ),
        "multi_tracklet_global_ids": (
            multi_tracklet_ids
        ),
        "alpha": args.alpha,
        "geometry_threshold_m": (
            args.geometry_threshold
        ),
        "acceptance_cost": (
            args.acceptance_cost
        ),
        "min_votes": args.min_votes,
        "min_support_ratio": (
            args.min_support_ratio
        ),
        "min_mean_similarity": (
            args.min_mean_similarity
        ),
        "max_median_distance_m": (
            args.max_median_distance
        ),
        "max_same_camera_overlap": (
            args.max_same_camera_overlap
        ),
    }

    with (
        args.output / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    pd.DataFrame([summary]).to_csv(
        args.output / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Готово.")
    print(
        f"Frame-level matches: "
        f"{len(frame_matches)}"
    )
    print(
        f"Accepted tracklet edges: "
        f"{accepted_edges}"
    )
    print(
        f"Global IDs: {global_id_count}"
    )
    print(
        f"Cross-camera global IDs: "
        f"{cross_camera_ids}"
    )
    print(
        f"Singleton global IDs: "
        f"{singleton_ids}"
    )
    print(
        "Results:",
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
