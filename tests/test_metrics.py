from mct_research.metrics import aggregate_run_metrics, build_track_summary


def test_track_summary_and_path_length() -> None:
    rows = [
        {
            "camera_id": "cam01", "frame_index": 0, "timestamp_s": 0.0,
            "track_id": 7, "class_id": 0, "class_name": "person",
            "confidence": 0.9, "x1": 0, "y1": 0, "x2": 2, "y2": 2,
            "center_x": 1, "center_y": 1,
        },
        {
            "camera_id": "cam01", "frame_index": 10, "timestamp_s": 1.0,
            "track_id": 7, "class_id": 0, "class_name": "person",
            "confidence": 0.8, "x1": 3, "y1": 4, "x2": 5, "y2": 6,
            "center_x": 4, "center_y": 5,
        },
    ]
    summary = build_track_summary(rows, fps=10.0)
    assert summary.iloc[0]["duration_s"] == 1.0
    assert summary.iloc[0]["path_length_px"] == 5.0


def test_aggregate_run_metrics() -> None:
    rows = [
        {"camera_id": "cam01", "track_id": 1},
        {"camera_id": "cam01", "track_id": 1},
        {"camera_id": "cam01", "track_id": 2},
    ]
    metrics = aggregate_run_metrics(rows, processed_frames=2, elapsed_seconds=1.0, fps=25)
    assert metrics["unique_tracks"] == 2
    assert metrics["processing_fps"] == 2.0
