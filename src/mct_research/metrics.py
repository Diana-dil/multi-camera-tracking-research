from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import pandas as pd


TRACK_COLUMNS = [
    "camera_id",
    "frame_index",
    "timestamp_s",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
]


def build_track_summary(rows: Iterable[dict], fps: float) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows), columns=TRACK_COLUMNS)
    columns = [
        "camera_id",
        "track_id",
        "class_name",
        "first_frame",
        "last_frame",
        "frames_seen",
        "duration_s",
        "mean_confidence",
        "path_length_px",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    summaries: list[dict] = []
    for (camera_id, track_id), group in frame.groupby(["camera_id", "track_id"]):
        ordered = group.sort_values("frame_index")
        points = list(zip(ordered["center_x"], ordered["center_y"]))
        path_length = sum(
            math.dist(points[index - 1], points[index])
            for index in range(1, len(points))
        )
        first_frame = int(ordered["frame_index"].min())
        last_frame = int(ordered["frame_index"].max())
        duration = (last_frame - first_frame) / fps if fps > 0 else 0.0
        summaries.append(
            {
                "camera_id": camera_id,
                "track_id": int(track_id),
                "class_name": str(ordered["class_name"].iloc[0]),
                "first_frame": first_frame,
                "last_frame": last_frame,
                "frames_seen": int(len(ordered)),
                "duration_s": round(float(duration), 4),
                "mean_confidence": round(float(ordered["confidence"].mean()), 4),
                "path_length_px": round(float(path_length), 2),
            }
        )
    return pd.DataFrame(summaries, columns=columns).sort_values(
        ["camera_id", "track_id"]
    )


def aggregate_run_metrics(
    rows: list[dict],
    processed_frames: int,
    elapsed_seconds: float,
    fps: float,
) -> dict[str, float | int]:
    unique_tracks = len({(row["camera_id"], row["track_id"]) for row in rows})
    detections = len(rows)
    processing_fps = processed_frames / elapsed_seconds if elapsed_seconds > 0 else 0.0
    video_seconds = processed_frames / fps if fps > 0 else 0.0
    return {
        "processed_frames": processed_frames,
        "video_seconds": round(video_seconds, 4),
        "elapsed_seconds": round(elapsed_seconds, 4),
        "processing_fps": round(processing_fps, 4),
        "unique_tracks": unique_tracks,
        "total_track_observations": detections,
        "mean_observations_per_frame": round(
            detections / processed_frames if processed_frames else 0.0, 4
        ),
    }
