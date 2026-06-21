from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a CVAT MOT 1.1 export into the project ground-truth CSV format."
    )
    parser.add_argument("--input", required=True, help="CVAT MOT zip, gt.txt, or MOT text file")
    parser.add_argument("--output", required=True, help="Output ground-truth CSV")
    parser.add_argument("--fps", type=float, required=True, help="Video FPS, e.g. 25")
    parser.add_argument("--camera-id", default="cam01")
    parser.add_argument(
        "--keep-zero-confidence",
        action="store_true",
        help="Keep rows whose MOT confidence/visibility field is 0",
    )
    return parser.parse_args()


def locate_mot_text(path: Path, temp_dir: Path) -> Path:
    if path.suffix.lower() != ".zip":
        return path
    with zipfile.ZipFile(path) as archive:
        candidates = [name for name in archive.namelist() if name.lower().endswith("gt.txt")]
        if not candidates:
            candidates = [name for name in archive.namelist() if name.lower().endswith(".txt")]
        if not candidates:
            raise FileNotFoundError("No gt.txt or MOT text file was found inside the CVAT export")
        selected = sorted(candidates, key=lambda value: ("/gt/" not in value.lower(), len(value)))[0]
        archive.extract(selected, temp_dir)
        return temp_dir / selected


def convert(input_path: Path, output_path: Path, fps: float, camera_id: str, keep_zero: bool) -> None:
    if fps <= 0:
        raise ValueError("fps must be positive")
    with tempfile.TemporaryDirectory() as directory:
        mot_path = locate_mot_text(input_path, Path(directory))
        mot = pd.read_csv(mot_path, header=None, comment="#")
    if mot.shape[1] < 6:
        raise ValueError("MOT file must contain at least 6 columns")

    names = ["frame", "track_id", "x", "y", "w", "h", "confidence", "x3d", "y3d", "z3d"]
    mot = mot.iloc[:, : min(10, mot.shape[1])].copy()
    mot.columns = names[: mot.shape[1]]
    if "confidence" not in mot.columns:
        mot["confidence"] = 1.0
    if not keep_zero:
        mot = mot[mot.confidence > 0]

    result = pd.DataFrame()
    result["camera_id"] = camera_id
    result["frame_index"] = mot.frame.astype(int) - 1
    result["timestamp_s"] = result.frame_index / fps
    result["track_id"] = mot.track_id.astype(int)
    result["class_id"] = 0
    result["class_name"] = "person"
    result["confidence"] = 1.0
    result["x1"] = mot.x.astype(float)
    result["y1"] = mot.y.astype(float)
    result["x2"] = mot.x.astype(float) + mot.w.astype(float)
    result["y2"] = mot.y.astype(float) + mot.h.astype(float)
    result["center_x"] = (result.x1 + result.x2) / 2.0
    result["center_y"] = (result.y1 + result.y2) / 2.0
    result = result.sort_values(["frame_index", "track_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(result)} ground-truth boxes to: {output_path}")
    print(f"Frames: {result.frame_index.min()}..{result.frame_index.max()}")
    print(f"Unique identities: {result.track_id.nunique()}")


def main() -> None:
    args = parse_args()
    convert(
        Path(args.input),
        Path(args.output),
        args.fps,
        args.camera_id,
        args.keep_zero_confidence,
    )


if __name__ == "__main__":
    main()
