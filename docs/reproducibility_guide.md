# E01 Reproducibility Guide

## 1. Зачем нужна эта структура

Эксперимент разбит на:

- один общий блок `foundation`, который готовит данные, split, ROI и признаки;
- пять отдельных блоков `H1-H5`, каждый из которых проверяет одну научную гипотезу;
- один общий блок `all`, который запускает все последовательно.

Это сделано для того, чтобы человек мог:

- пересобрать весь эксперимент целиком;
- отдельно запустить только одну гипотезу;
- быстро понять, какие артефакты относятся к какой гипотезе.

Дополнительно все новые entrypoint-скрипты работают в resumable-режиме:

- если артефакты уже есть, они не пересчитываются без необходимости;
- если чего-то не хватает, достраивается только отсутствующий этап;
- в summary каждого блока фиксируется, что именно было пересчитано в текущем запуске.

## 2. Основные команды

Рабочая директория:

- `/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline`

Перед запуском перейдите в каталог эксперимента:

```bash
cd /Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline
```

Перед запуском нужно подготовить окружение и указать пути к данным.

### 2.1. Установка зависимостей

Проект не хранит `.venv` в репозитории. После клонирования создайте окружение и установите зависимости:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.2. Где должны лежать тяжелые данные

В репозиторий не включаются:

- исходный датасет `data`;
- датасет масок `data_masks`;
- все артефакты из `outputs/`.

Ожидается, что у вас уже есть две директории:

- `data_root` — каталог с исходными изображениями и label-файлами;
- `masks_root` — каталог с масками объектов.

Проект не требует, чтобы эти каталоги лежали внутри репозитория. Они могут находиться в любом месте файловой системы.

### 2.3. Как указать пути к данным

Рекомендуемый способ:

1. Скопируйте [local_paths.template.json](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/configs/datasets/local_paths.template.json) в `configs/datasets/local_paths.json`.
2. Впишите абсолютные пути:

```json
{
  "data_root": "/absolute/path/to/Data/data",
  "masks_root": "/absolute/path/to/Data/data_masks"
}
```

Этот файл специально не коммитится.

Альтернативы:

- переменные среды:

```bash
export E01_DATA_ROOT="/absolute/path/to/Data/data"
export E01_MASKS_ROOT="/absolute/path/to/Data/data_masks"
```

- аргументы командной строки:

```bash
./.venv/bin/python scripts/run_all_hypotheses.py --data-root /absolute/path/to/Data/data --masks-root /absolute/path/to/Data/data_masks
```

- передача путей прямо через `make`:

```bash
make all DATA_ROOT=/absolute/path/to/Data/data MASKS_ROOT=/absolute/path/to/Data/data_masks
```

После этого можно использовать либо `make`, либо прямой вызов Python.

### Вариант A. Самый удобный: через `make`

```bash
make foundation
make h1
make h2
make h3
make h4
make h5
make all
make test
make repro-check
```

Если вы не хотите создавать `configs/datasets/local_paths.json`, можно сразу передать пути:

```bash
make foundation DATA_ROOT=/absolute/path/to/Data/data MASKS_ROOT=/absolute/path/to/Data/data_masks
make all DATA_ROOT=/absolute/path/to/Data/data MASKS_ROOT=/absolute/path/to/Data/data_masks
```

### Вариант B. Прямой запуск Python

```bash
./.venv/bin/python scripts/run_foundation.py
./.venv/bin/python scripts/run_h1_interpretable_features.py
./.venv/bin/python scripts/run_h2_mask_effect.py
./.venv/bin/python scripts/run_h3_noise_regimes.py
./.venv/bin/python scripts/run_h4_aggregation.py
./.venv/bin/python scripts/run_h5_deep_vs_handcrafted.py
./.venv/bin/python scripts/run_all_hypotheses.py
./.venv/bin/python -m unittest discover -s tests -v
```

Какие скрипты предназначены для прямого запуска человеком, а какие являются внутренними реализациями этапов, указано в [scripts/README.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/scripts/README.md).

## 3. Что запускает каждый блок

### `foundation`

Файл:

- [run_foundation.py](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/scripts/run_foundation.py)

Что делает:

- `A0`: аудит данных и индексы;
- `A1`: подмножества корпуса;
- `A2`: patient-level split;
- `A4`: постановки бинарных задач;
- `A3`: ROI по bbox и маскам;
- `A5`: извлечение признаков.

Главный смысл:

- один раз подготовить все базовые артефакты для всех следующих гипотез.

Что ожидается на выходе:

- `outputs/dataset_index/` — индексы пациентов, изображений и объектов;
- `outputs/splits/` — patient-level split;
- `outputs/crops/` — bbox и mask ROI;
- `outputs/features/` — таблицы признаков;
- `outputs/reports/` — QC-отчеты этапов `A0-A5`.

Ключевой summary:

- [foundation_summary.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/hypotheses/foundation_summary.md)

### `h1`

Файл:

- [run_h1_interpretable_features.py](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/scripts/run_h1_interpretable_features.py)

Что делает:

- `A6`: baseline-модели на признаках;
- `A7`: устойчивость и интерпретация признаков.

Проверяемая гипотеза:

- интерпретируемые признаки действительно помогают отличать бласты.

Что ожидается на выходе:

- `outputs/tables/object_level_metrics_baselines.csv`;
- `outputs/tables/baseline_model_leaderboard.csv`;
- `outputs/predictions/object_level_baseline_predictions.parquet`;
- `outputs/tables/top_features_stability.csv`;
- `outputs/figures/pr_curves_baselines.png`;
- `outputs/figures/roc_curves_baselines.png`;
- `outputs/reports/feature_interpretation_notes.md`.

Summary:

- [h1_interpretable_features_summary.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/hypotheses/h1_interpretable_features_summary.md)

### `h2`

Файл:

- [run_h2_mask_effect.py](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/scripts/run_h2_mask_effect.py)

Что делает:

- масочная часть `A8`.

Проверяемая гипотеза:

- маска клетки улучшает признаки и/или качество классификации.

Что ожидается на выходе:

- `outputs/tables/ablation_mask_usage.csv`.

Summary:

- [h2_mask_effect_summary.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/hypotheses/h2_mask_effect_summary.md)

### `h3`

Файл:

- [run_h3_noise_regimes.py](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/scripts/run_h3_noise_regimes.py)

Что делает:

- режимы корпуса и шумовая часть `A8`.

Проверяемая гипотеза:

- шумовые классы и более реалистичный режим корпуса усложняют задачу.

Что ожидается на выходе:

- `outputs/tables/ablation_dataset_regimes.csv`;
- `outputs/figures/ablation_summary.png` при совместном запуске с `H2` или через `all`.

Summary:

- [h3_noise_regimes_summary.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/hypotheses/h3_noise_regimes_summary.md)

### `h4`

Файл:

- [run_h4_aggregation.py](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/scripts/run_h4_aggregation.py)

Что делает:

- `A10-A11`: image-level и patient-level агрегирование, итоговые фигуры и таблицы.

Проверяемая гипотеза:

- object-level score можно превратить в practically useful оценку изображения и пациента.

Что ожидается на выходе:

- `outputs/tables/image_level_metrics.csv`;
- `outputs/tables/patient_level_ranking_metrics.csv`;
- `outputs/predictions/image_level_scores.parquet`;
- `outputs/predictions/patient_level_scores.parquet`;
- `outputs/tables/figure_manifest.csv`;
- `outputs/tables/table_manifest.csv`.

Summary:

- [h4_aggregation_summary.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/hypotheses/h4_aggregation_summary.md)

### `h5`

Файл:

- [run_h5_deep_vs_handcrafted.py](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/scripts/run_h5_deep_vs_handcrafted.py)

Что делает:

- подготовка `modeling_workset`;
- `A9`: ограниченные deep-baseline.

Проверяемая гипотеза:

- lightweight deep-модели можно честно сравнить с handcrafted baseline на тех же данных.

Что ожидается на выходе:

- `outputs/dataset_index/task_a43_full_realistic_modeling_workset.parquet`;
- `outputs/crops/roi_manifest_modeling_workset.parquet`;
- `outputs/tables/deep_vs_handcrafted.csv`;
- `outputs/predictions/object_level_deep_predictions.parquet`;
- `outputs/figures/deep_training_curves.png`;
- `outputs/reports/a9_deep_qc.md`.

Summary:

- [h5_deep_vs_handcrafted_summary.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/hypotheses/h5_deep_vs_handcrafted_summary.md)

### `all`

Файл:

- [run_all_hypotheses.py](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/scripts/run_all_hypotheses.py)

Что делает:

- последовательно запускает `foundation`, `H1`, `H2`, `H3`, `H4`, `H5`.

Summary:

- [run_all_hypotheses_summary.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/hypotheses/run_all_hypotheses_summary.md)

## 4. Порядок запуска по зависимостям

Если вы хотите выполнять эксперимент по частям, используйте такой порядок:

1. `foundation`
2. `h1`
3. `h2`
4. `h3`
5. `h4`
6. `h5`

Почему именно так:

- `H1-H5` зависят от базовых артефактов `foundation`;
- `H2`, `H3`, `H4`, `H5` используют результаты `H1`;
- `H5` дополнительно строит `modeling_workset`.

Если нужна полная пересборка без ручного контроля, используйте:

```bash
make all
```

или

```bash
./.venv/bin/python scripts/run_all_hypotheses.py
```

Если артефакты уже собраны, эта команда отработает быстро и просто переиспользует проверенные результаты.

## 5. Где смотреть результаты

Основные директории:

- [outputs/reports/hypotheses](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/hypotheses) — короткие summary по каждому блоку;
- [outputs/tables](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/tables) — численные таблицы метрик;
- [outputs/figures](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/figures) — итоговые графики;
- [outputs/predictions](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/predictions) — предсказания моделей;
- [outputs/checkpoints](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/checkpoints) — сохраненные модели.

Практически это означает:

- если нужен “сырой” численный результат — смотрите `outputs/tables/`;
- если нужны предсказания моделей — смотрите `outputs/predictions/`;
- если нужны изображения и графики для статьи или диссертации — смотрите `outputs/figures/`;
- если нужна проверка корректности этапов — смотрите `outputs/reports/` и `outputs/reports/hypotheses/`.

Подробный человекочитаемый отчет:

- [E01_detailed_report.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/reports/E01_detailed_report.md)
- [E01_detailed_report.docx](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/reports/E01_detailed_report.docx)
- [E01_detailed_report.pdf](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/reports/E01_detailed_report.pdf)

## 6. Как проверять, что все прошло корректно

Минимальная проверка:

```bash
make repro-check
```

Что это означает:

- полностью пересобран воспроизводимый запуск;
- после этого выполнены локальные тесты.

Проверенный сценарий публикационного репозитория:

1. Установить зависимости из `requirements.txt`.
2. Создать `configs/datasets/local_paths.json`.
3. Выполнить `make all`.
4. Выполнить `make test`.
5. Открыть `outputs/reports/hypotheses/run_all_hypotheses_summary.md`.

Также полезно открыть:

- [run_all_hypotheses_summary.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/hypotheses/run_all_hypotheses_summary.md)
- [experiment_execution_log.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/reports/experiment_execution_log.md)

## 7. Важное ограничение среды

В текущей среде настоящий `xgboost` не используется из-за отсутствия `libomp.dylib`.

Практическое следствие:

- конфигурации, которые в таблицах помечены как `xgboost`, фактически выполняются через `HistGradientBoosting` fallback.

Это ограничение уже отражено в:

- [a6_baseline_qc.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/a6_baseline_qc.md)
- [a9_deep_qc.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/a9_deep_qc.md)
- [experiment_execution_log.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/reports/experiment_execution_log.md)
