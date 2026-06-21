import pandas as pd

from mct_research.evaluation import evaluate_tracking


def rows(ids):
    result = []
    for frame_index, track_id in enumerate(ids):
        if track_id is None:
            continue
        result.append(
            {
                "frame_index": frame_index,
                "track_id": track_id,
                "x1": 0,
                "y1": 0,
                "x2": 10,
                "y2": 10,
            }
        )
    return pd.DataFrame(result)


def test_perfect_tracking():
    gt = rows([1, 1, 1])
    pred = rows([7, 7, 7])
    result = evaluate_tracking(gt, pred)
    assert result.metrics["mota"] == 1.0
    assert result.metrics["idf1"] == 1.0
    assert result.metrics["id_switches"] == 0


def test_id_switch_is_counted():
    gt = rows([1, 1])
    pred = rows([7, 8])
    result = evaluate_tracking(gt, pred)
    assert result.metrics["id_switches"] == 1
    assert result.metrics["idf1"] == 0.5


def test_missing_detection_is_false_negative():
    gt = rows([1, 1])
    pred = rows([7, None])
    result = evaluate_tracking(gt, pred)
    assert result.metrics["false_negatives"] == 1
    assert result.metrics["recall"] == 0.5
