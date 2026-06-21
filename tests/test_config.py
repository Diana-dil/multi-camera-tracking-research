from pathlib import Path

import pytest

from mct_research.config import TrackingConfig


def test_load_bytetrack_config() -> None:
    path = Path("configs/experiments/bytetrack.yaml")
    config = TrackingConfig.from_yaml(path)
    assert config.model == "yolo11n.pt"
    assert config.tracker == "bytetrack.yaml"
    assert config.classes == (0,)


def test_invalid_threshold_is_rejected() -> None:
    config = TrackingConfig(
        experiment_name="bad",
        camera_id="cam01",
        model="yolo11n.pt",
        tracker="bytetrack.yaml",
        source="video.mp4",
        confidence_threshold=1.1,
    )
    with pytest.raises(ValueError):
        config.validate()
