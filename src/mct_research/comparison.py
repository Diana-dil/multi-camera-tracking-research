from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


COMPARISON_COLUMNS = [
    "experiment_name",
    "model",
    "tracker",
    "source",
    "device",
    "processed_frames",
    "video_seconds",
    "elapsed_seconds",
    "processing_fps",
    "unique_tracks",
    "total_track_observations",
    "mean_observations_per_frame",
]


def find_summaries(root: str | Path) -> list[Path]:
    return sorted(Path(root).glob("**/summary.json"))


def load_comparison(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        metrics = payload.get("metrics", {})
        rows.append(
            {
                "experiment_name": payload.get("experiment_name"),
                "model": payload.get("model"),
                "tracker": payload.get("tracker"),
                "source": payload.get("source"),
                "device": payload.get("device"),
                **metrics,
            }
        )
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
