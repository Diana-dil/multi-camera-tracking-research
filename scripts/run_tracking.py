from __future__ import annotations

import argparse

from mct_research.config import TrackingConfig
from mct_research.tracking import run_tracking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible YOLO tracking experiment."
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    parser.add_argument("--source", help="Override source video path")
    parser.add_argument("--output-root", help="Override results directory")
    parser.add_argument("--max-frames", type=int, help="Process only the first N frames")
    parser.add_argument("--device", help="auto, cpu, 0, 1, ...")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrackingConfig.from_yaml(args.config).with_overrides(
        source=args.source,
        output_root=args.output_root,
        max_frames=args.max_frames,
        device=args.device,
    )
    run_tracking(config)


if __name__ == "__main__":
    main()
