from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrackingConfig:
    experiment_name: str
    camera_id: str
    model: str
    tracker: str
    source: str
    classes: tuple[int, ...] = (0,)
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.70
    image_size: int = 640
    device: str = "auto"
    save_video: bool = True
    save_csv: bool = True
    show_preview: bool = False
    max_frames: int | None = None
    output_root: str = "results"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrackingConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file does not exist: {path}")
        with path.open("r", encoding="utf-8") as file:
            raw: dict[str, Any] = yaml.safe_load(file) or {}

        if "classes" in raw:
            raw["classes"] = tuple(int(value) for value in raw["classes"])

        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must not be empty")
        if not self.camera_id.strip():
            raise ValueError("camera_id must not be empty")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive or null")
        if not self.classes:
            raise ValueError("classes must contain at least one class id")

    def with_overrides(
        self,
        *,
        source: str | None = None,
        output_root: str | None = None,
        max_frames: int | None = None,
        device: str | None = None,
    ) -> "TrackingConfig":
        changes: dict[str, Any] = {}
        if source is not None:
            changes["source"] = source
        if output_root is not None:
            changes["output_root"] = output_root
        if max_frames is not None:
            changes["max_frames"] = max_frames
        if device is not None:
            changes["device"] = device
        updated = replace(self, **changes)
        updated.validate()
        return updated

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["classes"] = list(self.classes)
        return data
