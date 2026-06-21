from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mct_research.evaluation import evaluate_tracking, write_motchallenge_file


def parse_prediction(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=path/to/observations.csv")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Prediction name and path must not be empty")
    return name.strip(), Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one or more tracker observation CSVs against ground truth."
    )
    parser.add_argument("--ground-truth", required=True, help="Ground-truth CSV")
    parser.add_argument(
        "--prediction",
        action="append",
        type=parse_prediction,
        required=True,
        help="Repeatable: ByteTrack=path/to/observations.csv",
    )
    parser.add_argument("--output", default="results/ground_truth_evaluation")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument(
        "--export-mot",
        action="store_true",
        help="Also save prediction and GT files in MOTChallenge text format",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    ground_truth = pd.read_csv(gt_path)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    if args.export_mot:
        write_motchallenge_file(ground_truth, output_root / "mot" / "gt.txt")

    for name, prediction_path in args.prediction:
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        predictions = pd.read_csv(prediction_path)
        result = evaluate_tracking(
            ground_truth,
            predictions,
            iou_threshold=args.iou_threshold,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
        run_dir = output_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump(result.metrics, file, indent=2, ensure_ascii=False)
        result.matches.to_csv(run_dir / "matches.csv", index=False, encoding="utf-8-sig")
        result.per_track.to_csv(run_dir / "per_track.csv", index=False, encoding="utf-8-sig")
        if args.export_mot:
            write_motchallenge_file(predictions, output_root / "mot" / f"{name}.txt")
        summary_rows.append({"tracker": name, **result.metrics})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "summary.csv", index=False, encoding="utf-8-sig")
    selected = [
        "tracker",
        "mota",
        "motp_iou",
        "idf1",
        "id_switches",
        "fragmentations",
        "precision",
        "recall",
        "false_positives",
        "false_negatives",
    ]
    print(summary[[column for column in selected if column in summary.columns]].to_string(index=False))
    print(f"\nDetailed results: {output_root.resolve()}")


if __name__ == "__main__":
    main()
