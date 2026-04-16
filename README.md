# E01 Textural Features Blast Pipeline

## Постановка задачи

Проверить, насколько интерпретируемые морфометрические, цветовые и текстурные признаки действительно полезны для автоматизированного обнаружения бластных клеток на микроскопических изображениях препаратов крови, если оценка проводится на реальных данных с разметкой объектов, масками и разделением по пациентам.

## Предмет проекта

- объектно-уровневая постановка `blast / non-blast`;
- сравнение признаковых и более сложных моделей на одинаковых данных;
- оценка роли сегментационных масок при извлечении признаков;
- переход от классификации отдельного объекта к анализаторно-ориентированной оценке изображения;
- фиксация ограничений текущих данных и выделение задач, требующих нового корпуса.

## Текущий фокус

На первом этапе приоритет отдается только тем частям исследования, которые можно выполнить на локальных данных.

- object detection-разметка по пациентам;
- автоматически построенные маски объектов;
- patient-level разбиение;
- морфо-текстурные признаки на уровне отдельных клеток;
- image-level агрегирование как приближение к задаче автоматического анализатора.

Постановки, требующие явного разделения лимфобластов и миелобластов, внешней межлабораторной валидации и case-level клинических меток, вынесены во вторую часть плана.

## Навигация по проекту

- [docs/data_audit.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/docs/data_audit.md) — зафиксированный аудит структуры и качества доступных данных.
- [plan/current/experiment_plan_detailed.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/plan/current/experiment_plan_detailed.md) — подробный рабочий план эксперимента.
- [docs/reproducibility_guide.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/docs/reproducibility_guide.md) — подробная инструкция по воспроизводимому запуску `foundation`, `H1-H5` и полного прогона.

## Воспроизводимый запуск

Эксперимент организован в виде:

- одного базового блока `foundation`;
- отдельных блоков `H1-H5` по гипотезам;
- общего блока `all` для полного последовательного прогона.

Новые entrypoint-скрипты работают в resumable-режиме:

- если нужные артефакты уже существуют, они валидируются и переиспользуются;
- если какого-то этапа не хватает, пересчитывается только недостающая часть;
- итоговые короткие summary по каждому блоку сохраняются в [outputs/reports/hypotheses](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/hypotheses).

## Что нужно перед первым запуском

1. Создать Python-окружение и установить зависимости из [requirements.txt](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/requirements.txt).
2. Скопировать [local_paths.template.json](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/configs/datasets/local_paths.template.json) в `configs/datasets/local_paths.json`.
3. Указать в `local_paths.json` абсолютные пути к:
   `data_root` — каталогу с исходным датасетом `data`;
   `masks_root` — каталогу с датасетом масок `data_masks`.
4. После этого выполнить `make all`.

Альтернатива конфигу:

- можно не создавать `local_paths.json`, а передавать пути через переменные среды `E01_DATA_ROOT` и `E01_MASKS_ROOT`;
- либо запускать [run_foundation.py](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/scripts/run_foundation.py) и [run_all_hypotheses.py](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/scripts/run_all_hypotheses.py) с аргументами `--data-root` и `--masks-root`.
- либо передавать пути прямо в `make`, например: `make all DATA_ROOT=/absolute/path/to/Data/data MASKS_ROOT=/absolute/path/to/Data/data_masks`.

Быстрые команды из каталога эксперимента:

```bash
make foundation
make h1
make h2
make h3
make h4
make h5
make all
make repro-check
```

Прямые entrypoint-скрипты:

- [run_foundation.py](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/scripts/run_foundation.py)
- [run_h1_interpretable_features.py](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/scripts/run_h1_interpretable_features.py)
- [run_h2_mask_effect.py](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/scripts/run_h2_mask_effect.py)
- [run_h3_noise_regimes.py](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/scripts/run_h3_noise_regimes.py)
- [run_h4_aggregation.py](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/scripts/run_h4_aggregation.py)
- [run_h5_deep_vs_handcrafted.py](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/scripts/run_h5_deep_vs_handcrafted.py)
- [run_all_hypotheses.py](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/scripts/run_all_hypotheses.py)

Какие скрипты считаются публичными точками входа, а какие внутренними служебными шагами, отдельно описано в [scripts/README.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/scripts/README.md).

## Каталоги проекта

- `configs/` — конфигурации запуска; локальные пути к данным задаются через `configs/datasets/local_paths.json` и не коммитятся.
- `data_access/` — пояснения по доступу к данным без хранения самих данных в репозитории.
- `docs/` — аналитические материалы и аудит данных.
- `notebooks/` — исследовательские ноутбуки.
- `outputs/dataset_index/` — манифесты пациентов, изображений и объектов.
- `outputs/splits/` — зафиксированные patient-level разбиения.
- `outputs/crops/` — извлеченные объектные кропы.
- `outputs/features/` — таблицы признаков.
- `outputs/checkpoints/` — обученные модели.
- `outputs/predictions/` — предсказания на object-level и image-level.
- `outputs/figures/` и `outputs/tables/` — артефакты для статьи и диссертации.
- `outputs/reports/` и `reports/` — промежуточные отчеты и итоговые summary-файлы.
- `src/` — основной код проекта.
- `tests/` — проверки корректности пайплайна.

## Что не попадает в GitHub

В [\.gitignore](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/.gitignore) вынесены:

- все содержимое `outputs/`;
- локальное окружение `.venv/` и кэш `.uv-cache/`;
- локальный файл путей `configs/datasets/local_paths.json`;
- служебные и бинарные директории вроде `vendor/`;
- генерируемые экспортные версии отчетов `reports/*.pdf`, `reports/*.docx`, `reports/*.html`.

## Ключевые рабочие решения

- Основная постановка для текущих данных: `blast vs non-blast`, а не `lymphoblast vs myeloblast`.
- Главная единица недопущения утечки: пациент.
- В базовый patient-level протокол входят только `122` каталогов с patient id; специальный каталог `part-1` пока рассматривается отдельно до уточнения его происхождения.
- Основной сопоставитель объектов: порядок строк в `images_labels/*.txt` и индекс `obj_XXXX` в имени маски.
- Первичная научная ценность текущего корпуса: воспроизводимая проверка полезности текстурных признаков, влияния масок и реалистичных условий data quality.
