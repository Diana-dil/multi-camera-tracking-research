# Ground-truth evaluation of ByteTrack and BoT-SORT

This stage converts the exploratory comparison into a reproducible MOT evaluation.

## 1. Annotate the short video in CVAT

1. Create a CVAT task and upload `data/samples/people.mp4`.
2. Create one label: `person`.
3. Use **Track mode**, not separate rectangles. Each real person must keep one annotation track ID.
4. Correct boxes at entry, exit, occlusion, and direction changes. CVAT interpolates boxes between keyframes.
5. For a faster pilot, annotate frames `0..149` first. Later extend to all `0..340` frames.
6. Export the task as **MOT 1.1**.

Rules for reliable ground truth:

- Do not give a new ID after a short occlusion if it is clearly the same person.
- Do not annotate reflections or pictures of people.
- Keep partially visible people if their location is unambiguous.
- Use one consistent rule for people at the extreme image border.

## 2. Convert the CVAT export

```powershell
python scripts/import_cvat_mot.py `
  --input data/annotations/people_cvat_mot.zip `
  --output data/annotations/people_gt.csv `
  --fps 25
```

CVAT/MOT uses 1-based frame numbers. The converter changes them to this project's 0-based `frame_index`.

## 3. Evaluate both trackers

```powershell
python scripts/evaluate_tracking.py `
  --ground-truth data/annotations/people_gt.csv `
  --prediction ByteTrack=results/exp_001_yolo11n_bytetrack/RUN/observations.csv `
  --prediction BoT-SORT=results/exp_002_yolo11n_botsort/RUN/observations.csv `
  --output results/ground_truth_evaluation `
  --iou-threshold 0.5 `
  --start-frame 0 `
  --end-frame 149 `
  --export-mot
```

Remove `--start-frame` and `--end-frame` after the whole video is annotated.

## 4. Outputs

```text
results/ground_truth_evaluation/
├── summary.csv
├── ByteTrack/
│   ├── metrics.json
│   ├── matches.csv
│   └── per_track.csv
├── BoT-SORT/
│   ├── metrics.json
│   ├── matches.csv
│   └── per_track.csv
└── mot/
    ├── gt.txt
    ├── ByteTrack.txt
    └── BoT-SORT.txt
```

The local evaluator reports:

- precision and recall;
- false positives and false negatives;
- MOTA;
- mean matched IoU (`motp_iou`);
- ID switches;
- fragmentations;
- ID precision, ID recall, and IDF1;
- mostly tracked and mostly lost trajectories.

## 5. Publication-grade evaluation

The local evaluator is transparent and useful for debugging. For final thesis tables, validate the exported MOT files with the official **TrackEval** implementation. TrackEval supports HOTA, CLEAR MOT metrics, and identity metrics. For a custom sequence, convert data to MOTChallenge layout and normally disable distractor preprocessing (`--DO_PREPROC False`).

## 6. Next multi-camera stage

After the single-camera metrics are obtained, download WILDTRACK from the official EPFL page. It contains seven synchronized HD cameras with largely overlapping fields of view, joint calibration, and annotations. Use only two cameras and a short frame range for the first multi-camera prototype.
