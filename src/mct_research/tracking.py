from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from .config import TrackingConfig
from .metrics import TRACK_COLUMNS, aggregate_run_metrics, build_track_summary
from .utils import environment_info, safe_name, timestamp_for_path, write_json, write_yaml


def resolve_device(requested: str) -> str:
    if requested.lower() != "auto":
        return requested
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _extract_rows(result: Any, camera_id: str, frame_index: int, fps: float) -> list[dict]:
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    track_ids = boxes.id.detach().cpu().numpy().astype(int)
    class_ids = boxes.cls.detach().cpu().numpy().astype(int)
    confidences = boxes.conf.detach().cpu().numpy()
    names = result.names

    rows: list[dict] = []
    for bbox, track_id, class_id, confidence in zip(
        xyxy, track_ids, class_ids, confidences, strict=True
    ):
        x1, y1, x2, y2 = (float(value) for value in bbox)
        rows.append(
            {
                "camera_id": camera_id,
                "frame_index": frame_index,
                "timestamp_s": round(frame_index / fps, 6) if fps > 0 else 0.0,
                "track_id": int(track_id),
                "class_id": int(class_id),
                "class_name": str(names.get(int(class_id), class_id)),
                "confidence": round(float(confidence), 6),
                "x1": round(x1, 3),
                "y1": round(y1, 3),
                "x2": round(x2, 3),
                "y2": round(y2, 3),
                "center_x": round((x1 + x2) / 2.0, 3),
                "center_y": round((y1 + y2) / 2.0, 3),
            }
        )
    return rows


def run_tracking(config: TrackingConfig) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc

    source = Path(config.source)
    if not source.exists():
        raise FileNotFoundError(
            f"Video not found: {source}. Put a video in data/samples or pass --source."
        )

    run_dir = (
        Path(config.output_root)
        / safe_name(config.experiment_name)
        / timestamp_for_path()
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    write_yaml(run_dir / "config.yaml", config.to_dict())

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {source}")

    input_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if input_fps <= 0:
        input_fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    writer: cv2.VideoWriter | None = None
    if config.save_video:
        video_path = run_dir / "annotated.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            input_fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create output video: {video_path}")

    model = YOLO(config.model)
    device = resolve_device(config.device)
    rows: list[dict] = []
    frame_index = 0
    started = time.perf_counter()

    try:
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            if config.max_frames is not None and frame_index >= config.max_frames:
                break

            result = model.track(
                source=frame,
                persist=True,
                tracker=config.tracker,
                classes=list(config.classes),
                conf=config.confidence_threshold,
                iou=config.iou_threshold,
                imgsz=config.image_size,
                device=device,
                verbose=False,
            )[0]

            rows.extend(_extract_rows(result, config.camera_id, frame_index, input_fps))
            annotated = result.plot()

            if writer is not None:
                writer.write(annotated)
            if config.show_preview:
                cv2.imshow("Tracking preview - press Q to stop", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    frame_index += 1
                    break

            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    observations = pd.DataFrame(rows, columns=TRACK_COLUMNS)
    track_summary = build_track_summary(rows, input_fps)

    if config.save_csv:
        observations.to_csv(run_dir / "observations.csv", index=False, encoding="utf-8")
        track_summary.to_csv(run_dir / "tracks_summary.csv", index=False, encoding="utf-8")

    metrics = aggregate_run_metrics(rows, frame_index, elapsed, input_fps)
    summary = {
        "experiment_name": config.experiment_name,
        "camera_id": config.camera_id,
        "model": config.model,
        "tracker": config.tracker,
        "source": str(source),
        "device": device,
        "video": {
            "fps": input_fps,
            "width": width,
            "height": height,
            "declared_total_frames": total_frames,
        },
        "metrics": metrics,
        "environment": environment_info(),
    }
    write_json(run_dir / "summary.json", summary)

    print(f"Experiment finished: {run_dir}")
    print(f"Processed frames: {metrics['processed_frames']}")
    print(f"Unique tracks: {metrics['unique_tracks']}")
    print(f"Processing FPS: {metrics['processing_fps']}")
    return run_dir
