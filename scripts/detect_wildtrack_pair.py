from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


DATASET_ROOT = Path("data/raw/WILDTRACK")
IMAGES_ROOT = DATASET_ROOT / "Image_subsets"
ANNOTATIONS_ROOT = DATASET_ROOT / "annotations_positions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Детекция людей YOLO на паре камер WILDTRACK "
            "и сопоставление детекций с ground truth по IoU."
        )
    )
    parser.add_argument("--camera-a", default="C1")
    parser.add_argument("--camera-b", default="C6")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.50,
        help="Минимальный IoU для сопоставления детекции с GT.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/wildtrack_C1_C6_yolo_50"),
    )
    return parser.parse_args()


def camera_to_view_number(camera_name: str) -> int:
    return int(camera_name.removeprefix("C")) - 1


def valid_view(view: dict) -> bool:
    x1 = float(view["xmin"])
    y1 = float(view["ymin"])
    x2 = float(view["xmax"])
    y2 = float(view["ymax"])
    return x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1


def load_gt_for_camera(
    annotation_path: Path,
    camera_name: str,
) -> list[dict]:
    with annotation_path.open("r", encoding="utf-8") as file:
        annotations = json.load(file)

    view_number = camera_to_view_number(camera_name)
    rows: list[dict] = []

    for person in annotations:
        person_id = int(person["personID"])
        position_id = int(person["positionID"])

        views = {
            int(view["viewNum"]): view
            for view in person.get("views", [])
        }

        view = views.get(view_number)

        if view is None or not valid_view(view):
            continue

        rows.append(
            {
                "person_id": person_id,
                "position_id": position_id,
                "bbox": np.array(
                    [
                        float(view["xmin"]),
                        float(view["ymin"]),
                        float(view["xmax"]),
                        float(view["ymax"]),
                    ],
                    dtype=np.float64,
                ),
            }
        )

    return rows


def pairwise_iou(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros(
            (len(boxes_a), len(boxes_b)),
            dtype=np.float64,
        )

    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = np.maximum(
        0.0,
        (a[..., 2] - a[..., 0])
        * (a[..., 3] - a[..., 1]),
    )
    area_b = np.maximum(
        0.0,
        (b[..., 2] - b[..., 0])
        * (b[..., 3] - b[..., 1]),
    )

    union = area_a + area_b - intersection

    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def match_predictions_to_gt(
    pred_boxes: np.ndarray,
    gt_rows: list[dict],
    threshold: float,
) -> tuple[dict[int, tuple[int, float]], int, int, int]:
    """
    Возвращает:
      pred_index -> (gt_index, IoU),
      TP, FP, FN.
    """
    if len(pred_boxes) == 0:
        return {}, 0, 0, len(gt_rows)

    if not gt_rows:
        return {}, 0, len(pred_boxes), 0

    gt_boxes = np.stack(
        [row["bbox"] for row in gt_rows],
        axis=0,
    )

    iou_matrix = pairwise_iou(pred_boxes, gt_boxes)

    pred_indices, gt_indices = linear_sum_assignment(
        1.0 - iou_matrix
    )

    matches: dict[int, tuple[int, float]] = {}

    for pred_index, gt_index in zip(
        pred_indices,
        gt_indices,
    ):
        value = float(iou_matrix[pred_index, gt_index])

        if value >= threshold:
            matches[int(pred_index)] = (
                int(gt_index),
                value,
            )

    true_positive = len(matches)
    false_positive = len(pred_boxes) - true_positive
    false_negative = len(gt_rows) - true_positive

    return (
        matches,
        true_positive,
        false_positive,
        false_negative,
    )


def clip_box(
    box: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.tolist()

    x1 = max(0, min(image_width - 1, int(round(x1))))
    y1 = max(0, min(image_height - 1, int(round(y1))))
    x2 = max(1, min(image_width, int(round(x2))))
    y2 = max(1, min(image_height, int(round(y2))))

    return x1, y1, x2, y2


def main() -> None:
    args = parse_args()

    annotation_files = sorted(
        ANNOTATIONS_ROOT.glob("*.json")
    )[: args.frames]

    if not annotation_files:
        raise FileNotFoundError(
            f"Аннотации не найдены: {ANNOTATIONS_ROOT.resolve()}"
        )

    cameras = [args.camera_a, args.camera_b]

    device: str | int = 0 if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Frames: {len(annotation_files)}")
    print(f"Cameras: {', '.join(cameras)}")

    model = YOLO(args.model)

    args.output.mkdir(parents=True, exist_ok=True)

    crops_root = args.output / "crops"

    for camera in cameras:
        (crops_root / camera).mkdir(
            parents=True,
            exist_ok=True,
        )

    detection_rows: list[dict] = []
    metric_rows: list[dict] = []

    totals = {
        camera: {
            "gt": 0,
            "pred": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for camera in cameras
    }

    total_jobs = len(annotation_files) * len(cameras)
    completed = 0

    for annotation_path in annotation_files:
        frame_name = annotation_path.stem
        frame_number = int(frame_name)

        for camera in cameras:
            image_path = (
                IMAGES_ROOT
                / camera
                / f"{frame_name}.png"
            )

            image = cv2.imread(str(image_path))

            if image is None:
                raise FileNotFoundError(
                    f"Не удалось прочитать: {image_path.resolve()}"
                )

            result = model.predict(
                source=image,
                classes=[0],
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )[0]

            if result.boxes is None or len(result.boxes) == 0:
                pred_boxes = np.empty((0, 4), dtype=np.float64)
                pred_scores = np.empty((0,), dtype=np.float64)
            else:
                pred_boxes = (
                    result.boxes.xyxy
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                pred_scores = (
                    result.boxes.conf
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )

            gt_rows = load_gt_for_camera(
                annotation_path=annotation_path,
                camera_name=camera,
            )

            matches, tp, fp, fn = match_predictions_to_gt(
                pred_boxes=pred_boxes,
                gt_rows=gt_rows,
                threshold=args.match_iou,
            )

            totals[camera]["gt"] += len(gt_rows)
            totals[camera]["pred"] += len(pred_boxes)
            totals[camera]["tp"] += tp
            totals[camera]["fp"] += fp
            totals[camera]["fn"] += fn

            metric_rows.append(
                {
                    "frame": frame_number,
                    "camera_id": camera,
                    "gt_count": len(gt_rows),
                    "prediction_count": len(pred_boxes),
                    "true_positive": tp,
                    "false_positive": fp,
                    "false_negative": fn,
                }
            )

            image_height, image_width = image.shape[:2]

            for pred_index, (
                box,
                confidence,
            ) in enumerate(
                zip(pred_boxes, pred_scores)
            ):
                x1, y1, x2, y2 = clip_box(
                    box,
                    image_width=image_width,
                    image_height=image_height,
                )

                crop_relative_path = (
                    Path("crops")
                    / camera
                    / (
                        f"{frame_name}_det_"
                        f"{pred_index:03d}.jpg"
                    )
                )

                crop_path = args.output / crop_relative_path
                crop = image[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                if not cv2.imwrite(
                    str(crop_path),
                    crop,
                ):
                    raise RuntimeError(
                        f"Не удалось сохранить: {crop_path.resolve()}"
                    )

                matched_gt_index = None
                matched_iou = 0.0
                gt_person_id = -1
                gt_position_id = -1

                if pred_index in matches:
                    matched_gt_index, matched_iou = (
                        matches[pred_index]
                    )

                    gt_row = gt_rows[matched_gt_index]
                    gt_person_id = int(
                        gt_row["person_id"]
                    )
                    gt_position_id = int(
                        gt_row["position_id"]
                    )

                detection_rows.append(
                    {
                        "frame": frame_number,
                        "camera_id": camera,
                        "detection_id": pred_index,
                        "confidence": float(confidence),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "foot_u": (x1 + x2) / 2.0,
                        "foot_v": float(y2),
                        "matched_gt": int(
                            matched_gt_index is not None
                        ),
                        "gt_person_id": gt_person_id,
                        "gt_position_id": gt_position_id,
                        "match_iou": float(matched_iou),
                        "crop_path": crop_relative_path.as_posix(),
                    }
                )

            completed += 1
            print(
                f"Processed: {completed}/{total_jobs}",
                end="\r",
            )

    print()

    detections = pd.DataFrame(detection_rows)
    per_frame_metrics = pd.DataFrame(metric_rows)

    detections.to_csv(
        args.output / "detections.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_frame_metrics.to_csv(
        args.output / "per_frame_detection_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows: list[dict] = []

    for camera in cameras:
        values = totals[camera]

        precision = (
            values["tp"]
            / (values["tp"] + values["fp"])
            if values["tp"] + values["fp"] > 0
            else 0.0
        )

        recall = (
            values["tp"]
            / (values["tp"] + values["fn"])
            if values["tp"] + values["fn"] > 0
            else 0.0
        )

        summary_rows.append(
            {
                "camera_id": camera,
                "frames": len(annotation_files),
                "gt_detections": values["gt"],
                "pred_detections": values["pred"],
                "true_positive": values["tp"],
                "false_positive": values["fp"],
                "false_negative": values["fn"],
                "precision": precision,
                "recall": recall,
            }
        )

    total_values = {
        key: sum(
            totals[camera][key]
            for camera in cameras
        )
        for key in ["gt", "pred", "tp", "fp", "fn"]
    }

    total_precision = (
        total_values["tp"]
        / (total_values["tp"] + total_values["fp"])
        if total_values["tp"] + total_values["fp"] > 0
        else 0.0
    )

    total_recall = (
        total_values["tp"]
        / (total_values["tp"] + total_values["fn"])
        if total_values["tp"] + total_values["fn"] > 0
        else 0.0
    )

    summary_rows.append(
        {
            "camera_id": "ALL",
            "frames": len(annotation_files) * len(cameras),
            "gt_detections": total_values["gt"],
            "pred_detections": total_values["pred"],
            "true_positive": total_values["tp"],
            "false_positive": total_values["fp"],
            "false_negative": total_values["fn"],
            "precision": total_precision,
            "recall": total_recall,
        }
    )

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        args.output / "detection_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    config = {
        "camera_a": args.camera_a,
        "camera_b": args.camera_b,
        "frames": len(annotation_files),
        "model": args.model,
        "imgsz": args.imgsz,
        "confidence_threshold": args.conf,
        "nms_iou_threshold": args.iou,
        "gt_match_iou_threshold": args.match_iou,
        "device": str(device),
    }

    with (
        args.output / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(summary.round(4).to_string(index=False))
    print()
    print(f"Результаты: {args.output.resolve()}")


if __name__ == "__main__":
    main()
