# Журнал выполнения E01

## 2026-04-09

- Инициализация журнала.
- Прочитан актуальный план `plan/current/experiment_plan_detailed.md`.
- Собрана карта проекта и артефактов `E01_textural_features_blast_pipeline`.
- Предварительное наблюдение: этапы `A0-A8`, `A10-A11` частично или полностью представлены существующими артефактами; этап `A9` по выходам выглядит незавершенным.
- Предварительное наблюдение: для `A5` обнаружены артефакты вида `*_lite`, поэтому требуется отдельная проверка на соответствие полному плану.
- Исправлена проблема воспроизводимости окружения: `src/dataset/build_index.py` и `tests/test_dataset_index.py` брали `pyarrow` из `vendor`, несовместимого с Python `3.12`; приоритет импорта изменен в пользу `.venv/site-packages`.
- Локальные тесты `./.venv/bin/python -m unittest discover -s tests -v` после исправления проходят успешно: `5/5 OK`.
- Этап `A0` пересобран через `./.venv/bin/python src/dataset/build_index.py`.
- Проверка `A0`: все ожидаемые счетчики в `outputs/reports/data_audit_runtime.md` имеют статус `MATCH`; `main_objects = 86396`, дублей по ключу `patient_id + image_id + object_id_within_image` нет, пропусков в `patient_id`, `class_name` и bbox-полях нет, `part-1` исключен из основного протокола.
- Этапы `A1`, `A2`, `A4` пересобраны через `./.venv/bin/python scripts/run_a1_to_a4.py`.
- Дополнительно исправлен численный дефект QC-отчетов: уникальные изображения в `A1`, `A2`, `A3` теперь считаются по паре `patient_id + image_id`, а не только по `image_id`.
- Проверка `A1`: сформированы подмножества `subset_full_realistic = 86376`, `subset_clean_cell = 78498`, `subset_strict_morphology = 74901`, `subset_mask_ready = 86250`; запрещенные классы корректно исключены из `clean` и `strict`; все объекты `mask_ready` имеют `has_mask = True`.
- Проверка `A2`: сформирован воспроизводимый `split_v1` с разбиением пациентов `85 / 18 / 19`; пересечения между `train`, `validation`, `test` отсутствуют; `part-1` в split-файлы не попал.
- Проверка `A4`: во всех трех задачах `target_binary` строго совпадает с правилом `class_id == 7`; все задачи содержат покрытия `train`, `val`, `test`.
- Текущее состояние после проверки: `A0`, `A1`, `A2`, `A4` соответствуют критериям плана; следующим шагом требуется проверить `A3` и довести `A5` с workset-версии до полного корпуса.

## 2026-04-10

- Проверка `A3` выполнена по существующим артефактам полного корпуса: `bbox_roi_manifest.parquet` и `roi_manifest.parquet` содержат `86376` строк; `mask_crop_path` заполнен для `86250` объектов; `mask_tight_crop_path` заполнен для `86166` объектов. Критерии покрытия ROI выполнены.
- Полный `A5` запущен отдельно на готовом `outputs/crops/roi_manifest.parquet`, без повторного пересчета ROI.
- Проверка `A5`: построены `features_bbox.parquet`, `features_mask.parquet`, `features_combined.parquet`; число строк совпадает с ожидаемым (`86376 / 86250 / 86376`); словарь признаков расширен до колонок `matrix`, `feature_name`, `feature_family`, `source_region`, `description`; файл `feature_missingness.csv` теперь содержит агрегированную информацию по `bbox`, `mask`, `combined`. Константных признаков в `combined` нет; признаки с пропусками явно зафиксированы.
- Этапы `A6-A8`, `A10-A11` пересчитаны на полных feature-артефактах через `./.venv/bin/python scripts/run_a6_to_a11.py`.
- Проверка `A6`: получено `105` запусков (`3 tasks * 5 models * 7 feature groups`), сохранено `537005` object-level предсказаний, создано `105` checkpoint-файлов признаковых моделей. Лучший run по `full_realistic` — `a43_full_realistic_binary__xgboost__full`, но фактическая реализация в этом окружении — `hist_gradient_boosting_fallback`.
- Проверка `A7`: обновлены `top_features_stability.csv`, `shap_summary_top_model.png`, `feature_importance_stability.png`, `feature_interpretation_notes.md`.
- Проверка `A8`: обновлены `ablation_mask_usage.csv`, `ablation_dataset_regimes.csv`, `ablation_summary.png`; обязательные абляции по маскам и режимам корпуса присутствуют.
- Проверка `A10`: обновлены `image_level_metrics.csv`, `patient_level_ranking_metrics.csv`, `image_level_scores.parquet`, `patient_level_scores.parquet`; все четыре способа агрегирования (`max_object_score`, `mean_top3_scores`, `count_objects_above_threshold`, `soft_noisy_or`) присутствуют на image-level и patient-level.
- Проверка `A11`: полный комплект минимальных фигур и таблиц существует; `figure_manifest.csv` содержит `9` строк, `table_manifest.csv` содержит `19` строк; все обязательные артефакты присутствуют.
- `A9` сначала завершился ошибкой: в табличную ветку late-fusion попадали строковые поля ROI-манифеста. Исправление: выбор tabular-колонок ограничен числовыми признаками из `feature_df`.
- `A9` повторно завершился ошибкой: late-fusion выдавал `NaN`-скоры из-за пропусков и неограниченных значений в табличных признаках. Исправление: добавлены imputation + z-score нормализация в `scripts/run_a9_deep_baselines.py` и `nan_to_num`-защита в `src/models/deep_baselines.py`.
- После исправлений `A9` выполнен успешно: созданы `deep_vs_handcrafted.csv`, `object_level_deep_predictions.parquet`, `cnn_bbox_best.pt`, `cnn_mask_best.pt`, `cnn_late_fusion_best.pt`, `cnn_training_history.json`, `deep_training_curves.png`, `a9_deep_qc.md`.
- Дополнительно нормализована структура `deep_vs_handcrafted.csv`: handcrafted и deep строки приведены к единой схеме колонок для прямого сравнения.
- Итог по `A9`: на modeling workset лучший test PR-AUC у `handcrafted_best_full_realistic = 0.9390`; лучший deep-результат у `cnn_late_fusion = 0.8931`, что делает late fusion сильнейшей deep-базой, но не лучше лучшего handcrafted baseline.
- Финальная системная проверка: все обязательные артефакты этапов `A0-A11` существуют; пустой список `missing_required_files = []`.
- Зафиксированное ограничение окружения: реальный `xgboost` в `.venv` не поднимается из-за отсутствующего `libomp.dylib`, поэтому в `A6`/`A9` сравнение использует `HistGradientBoosting` fallback там, где код маркирует модель как `xgboost`. Это отражено в журналах и QC-отчетах.
- Для чтения человеком подготовлен подробный отчет эксперимента: `reports/E01_detailed_report.md`. В отчете простым языком описаны гипотезы, данные, методы, метрики, ключевые результаты, выводы и ссылки на основные артефакты и графики.
- Раздел `2. Какие гипотезы проверялись` в `reports/E01_detailed_report.md` дополнен: для каждой гипотезы подробно описаны смысл, способ проверки, используемые модели и критерии подтверждения.
- Для экспортов добавлен служебный рендерер `scripts/render_markdown_report_html.py`, который преобразует markdown-отчет в HTML с таблицами, ссылками и встроенными иллюстрациями.
- Созданы дополнительные человекочитаемые версии отчета: `reports/E01_detailed_report.docx` и `reports/E01_detailed_report.pdf`.
- Проверка экспортов: `E01_detailed_report.pdf` имеет `38` страниц формата `A4`; изображения встроены в PDF и DOCX; первая страница и страницы с графиками визуально проверены после рендера.
- Код эксперимента реорганизован в воспроизводимую структуру `foundation + H1-H5 + all`: добавлены `scripts/run_foundation.py`, `scripts/run_h1_interpretable_features.py`, `scripts/run_h2_mask_effect.py`, `scripts/run_h3_noise_regimes.py`, `scripts/run_h4_aggregation.py`, `scripts/run_h5_deep_vs_handcrafted.py`, `scripts/run_all_hypotheses.py`, общий orchestration-модуль `scripts/reproducibility_blocks.py`, `Makefile` и инструкция `docs/reproducibility_guide.md`.
- Новый orchestration работает в resumable-режиме: при наличии валидных артефактов шаги переиспользуются, а не пересчитываются заново; если чего-то не хватает, достраивается только отсутствующий этап. Для каждого блока пишутся summary-файлы в `outputs/reports/hypotheses`.
- Верификация новой структуры выполнена успешными запусками `./.venv/bin/python scripts/run_all_hypotheses.py`, `make all` и `make test`; summary-артефакты по блокам `foundation`, `H1-H5` и `all` созданы.

## 2026-04-11

- Проект подготовлен к публикации на GitHub: добавлены [`.gitignore`](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/.gitignore), [requirements.txt](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/requirements.txt), шаблон локальной конфигурации путей [local_paths.template.json](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/configs/datasets/local_paths.template.json) и пояснение по данным [data_access/README.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/data_access/README.md).
- Добавлен модуль [dataset_paths.py](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/src/utils/dataset_paths.py): пути к `data` и `data_masks` теперь можно задавать через локальный JSON-файл, переменные среды или аргументы CLI; жесткая привязка к локальному расположению данных снята.
- Обновлены [run_foundation.py](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/scripts/run_foundation.py), [run_all_hypotheses.py](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/scripts/run_all_hypotheses.py) и [reproducibility_blocks.py](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/scripts/reproducibility_blocks.py): summary явно фиксирует источник путей к данным и используемые каталоги.
- Выполнена очистка локального мусора перед публикацией: удалены `.DS_Store`, локальные `__pycache__`, HTML-экспорт отчета, локальная символьная ссылка на shared data, неиспользуемый runner `scripts/run_a6_postprocess_a7_to_a11.py`, а также локальные каталоги `.uv-cache/` и `vendor/`.
- Для снижения путаницы добавлен [scripts/README.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/scripts/README.md) с разделением на публичные точки входа и внутренние служебные скрипты этапов `A*`.
- `Makefile` расширен поддержкой передачи путей к данным напрямую: теперь возможны вызовы вида `make all DATA_ROOT=... MASKS_ROOT=...`; также добавлен target `clean-pyc` для повторной очистки Python-кэшей.
- Обновлены пользовательские инструкции в [README.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/README.md) и [docs/reproducibility_guide.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/docs/reproducibility_guide.md): подробно описаны размещение тяжелых данных, способы задания путей, запуск через `make` и ожидаемые артефакты.
- Проверка после GitHub-подготовки пройдена успешно: `./.venv/bin/python -m py_compile src/utils/dataset_paths.py scripts/reproducibility_blocks.py scripts/run_foundation.py scripts/run_all_hypotheses.py`, затем `./.venv/bin/python scripts/run_all_hypotheses.py --paths-config configs/datasets/local_paths.json`, затем `make all` и `make test`. Все команды завершились с кодом `0`.
- Дополнительно проверен новый интерфейс `Makefile`: `make all PATHS_CONFIG=configs/datasets/local_paths.json` завершился успешно и корректно передал путь конфигурации в `scripts/run_all_hypotheses.py`.
