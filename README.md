# Multi-Camera Tracking Research

Исследовательский проект для магистерской работы:

> «Разработка систем идентификации и трекинга подвижных объектов в системе видеонаблюдения с перекрывающимися полями обзора».

Текущая версия реализует **первый воспроизводимый baseline**: детекция людей YOLO11 и однокамерный трекинг с ByteTrack или BoT-SORT. Межкамерный ReID и объединение траекторий будут добавлены следующим этапом.

## Что уже реализовано

- конфигурации экспериментов в YAML;
- YOLO11n + ByteTrack;
- YOLO11n + BoT-SORT;
- обработка видео покадрово;
- фильтрация только класса `person`;
- сохранение видео с рамками и ID;
- сохранение всех наблюдений в CSV;
- агрегирование локальных траекторий;
- сохранение конфигурации и JSON-резюме запуска;
- сравнение нескольких запусков;
- тесты для конфигурации и расчёта базовых показателей;
- шаблон промежуточного отчёта.

## Требования

Рекомендуется:
- Windows 10/11;
- Python 3.11;
- 8 ГБ оперативной памяти или больше;
- NVIDIA GPU желателен, но первый эксперимент можно запустить на CPU.

## Установка на Windows

Откройте PowerShell в папке проекта:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1
```

Для NVIDIA GPU сначала установите подходящую сборку PyTorch по официальной инструкции PyTorch, затем выполните установку остальных зависимостей.

Ручная установка:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python scripts/check_environment.py
```

## Подготовка видео

Положите короткое видео с людьми сюда:

```text
data/samples/people.mp4
```

Или укажите путь через `--source`.

## Первый запуск: ByteTrack

```powershell
python scripts/run_tracking.py `
  --config configs/experiments/bytetrack.yaml `
  --max-frames 500
```

Полный запуск без ограничения кадров:

```powershell
python scripts/run_tracking.py --config configs/experiments/bytetrack.yaml
```

## Второй запуск: BoT-SORT

```powershell
python scripts/run_tracking.py `
  --config configs/experiments/botsort.yaml `
  --max-frames 500
```

## Запуск с видео из другой папки

```powershell
python scripts/run_tracking.py `
  --config configs/experiments/bytetrack.yaml `
  --source "C:\Users\Diana\Videos\people.mp4"
```

## Результаты эксперимента

Каждый запуск создаёт отдельную папку:

```text
results/
└── exp_001_yolo11n_bytetrack/
    └── 20260620_210000/
        ├── annotated.mp4
        ├── observations.csv
        ├── tracks_summary.csv
        ├── config.yaml
        └── summary.json
```

`observations.csv` содержит запись для каждого обнаруженного трека в каждом кадре.  
`tracks_summary.csv` содержит длительность и агрегированные характеристики каждого ID.  
`summary.json` содержит параметры видео и общие показатели запуска.

## Сравнение запусков

После выполнения обоих экспериментов:

```powershell
python scripts/compare_runs.py
```

Будут созданы:

```text
results/comparison.csv
results/comparison.md
```

## Тесты

```powershell
pytest
```

## Важное ограничение первого baseline

Скрипт пока не вычисляет HOTA, IDF1 и MOTA, потому что для них нужна эталонная покадровая разметка. Автоматически собираемые характеристики предназначены для технической проверки и предварительного сравнения. Для научного вывода они дополняются ручным подсчётом ID switches либо оценкой на MOT17/WILDTRACK.

## Следующий этап проекта

1. Модуль OSNet для ReID-признаков.
2. Усреднение признаков внутри локального tracklet.
3. Обработка двух синхронизированных камер.
4. Матрица межкамерного сходства.
5. Временной gating и ограничения зон входа/выхода.
6. Hungarian matching и назначение global ID.
7. Оценка на WILDTRACK и собственном пилотном наборе.

## Ground-truth tracking metrics

After the baseline runs, annotate `people.mp4` in CVAT Track mode and export MOT 1.1. Then use:

```powershell
python scripts/import_cvat_mot.py --input data/annotations/people_cvat_mot.zip --output data/annotations/people_gt.csv --fps 25
python scripts/evaluate_tracking.py --ground-truth data/annotations/people_gt.csv --prediction ByteTrack=PATH_TO_BYTE/observations.csv --prediction BoT-SORT=PATH_TO_BOT/observations.csv --output results/ground_truth_evaluation --export-mot
```

See `docs/ground_truth_evaluation.md` for the complete annotation and evaluation protocol.
