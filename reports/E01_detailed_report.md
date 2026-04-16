# Подробный отчет по эксперименту E01

## 1. Что это за эксперимент и зачем он нужен

Этот эксперимент проверяет очень практичный вопрос:

можно ли по изображению отдельной клетки крови автоматически понять, является ли она бластной клеткой (`blast`) или нет, если использовать не только "сырые картинки", но и понятные человеку признаки формы, цвета и текстуры.

Если говорить совсем просто, идея такая:

- у разных клеток разная форма;
- у них по-разному распределен цвет;
- у них по-разному выглядит внутренняя "текстура" изображения;
- все это можно перевести в числа и дать модели машинного обучения.

Главная цель эксперимента была сформулирована в плане так:

компактный и интерпретируемый набор морфометрических, цветовых и текстурных признаков должен устойчиво различать `blast / non-blast` на данных разных пациентов и быть полезным не только для отдельной клетки, но и для оценки целого изображения как приближения к задаче автоматического анализатора.

Исходный подробный план эксперимента находится здесь:

- [experiment_plan_detailed.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/plan/current/experiment_plan_detailed.md)

Журнал того, что реально делалось по шагам, находится здесь:

- [experiment_execution_log.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/reports/experiment_execution_log.md)

## 2. Какие гипотезы проверялись

В рамках эксперимента проверялись не одна, а сразу несколько связанных гипотез. Ниже каждая гипотеза сформулирована простым языком, а затем объясняется, как именно она проверялась на практике.

### Гипотеза 1. Признаки формы, цвета и текстуры действительно помогают отличать бласт от небласта

Смысл гипотезы:

если клетку описать не только картинкой, но и набором понятных чисел, то по этим числам можно надежно различать `blast / non-blast`.

Под "понятными числами" здесь имеются в виду:

- признаки формы;
- признаки цвета;
- признаки текстуры.

Почему это важно:

- такой подход легче объяснить врачу и исследователю;
- можно понять, какие именно свойства клетки помогают модели;
- это особенно полезно для диссертации, потому что результат получается интерпретируемым, а не "черным ящиком".

Как гипотеза проверялась:

1. Для каждой клетки были вычислены морфометрические, цветовые и текстурные признаки.
2. Далее обучались несколько признаковых моделей:
   - `Logistic Regression`;
   - `Linear SVM`;
   - `RBF SVM`;
   - `Random Forest`;
   - градиентный бустинг, который в таблицах обозначен как `xgboost`, но в данной среде фактически работал как `HistGradientBoosting` fallback.
3. Каждая модель обучалась не на одном, а на нескольких вариантах признаков:
   - только морфометрия;
   - только цвет;
   - только текстура;
   - морфометрия + цвет;
   - морфометрия + текстура;
   - цвет + текстура;
   - полный набор признаков.
4. Проверка проводилась на трех задачах:
   - `A4.1 clean_cell`;
   - `A4.2 strict_morphology`;
   - `A4.3 full_realistic`.

Что будет считаться подтверждением гипотезы:

- если хотя бы несколько моделей на patient-level split покажут устойчиво высокие метрики;
- если полный набор признаков или его сильные комбинации будут работать лучше простых вариантов;
- если полезность текстурных признаков будет видна не только в одной случайной модели, а в серии сравнений.

### Гипотеза 2. Маска клетки улучшает качество признаков и качество модели

Смысл гипотезы:

если точно отделить клетку от фона, то признаки будут чище, а значит и классификация станет лучше.

Это особенно логично для текстурных и цветовых признаков:

- если в bbox попадает фон;
- если рядом попадает другая клетка;
- если есть мусор или артефакт,

то числовые признаки уже описывают не только саму клетку, но и лишние объекты.

Как гипотеза проверялась:

1. Для каждого объекта были построены три вида ROI:
   - `bbox_crop`;
   - `mask_crop`;
   - `mask_tight_crop`.
2. Затем признаки считались в трех режимах:
   - только по `bbox`;
   - только по `mask`;
   - по объединению `bbox + mask`.
3. После этого на одинаковых split и одинаковых моделях сравнивались результаты:
   - `bbox_only`;
   - `mask_only`;
   - `bbox_plus_mask`.
4. Для этого использовался блок абляций `A8`, а сами модели брались из лучшего признакового family для соответствующей задачи.

Какие модели участвовали в проверке:

- главным образом лучшая boosting-модель для каждой задачи, потому что именно она использовалась для сравнений масок;
- дополнительно роль маски косвенно проверялась и в deep-блоке:
   - `cnn_bbox`;
   - `cnn_mask`;
   - `cnn_late_fusion`.

Что будет считаться подтверждением гипотезы:

- если `mask_only` работает не хуже `bbox_only` или лучше;
- если `bbox_plus_mask` оказывается лучше одного только `bbox`;
- если эффект наблюдается не на одной задаче, а повторяется хотя бы в нескольких режимах корпуса.

### Гипотеза 3. Шумовые классы действительно усложняют задачу

Смысл гипотезы:

если убрать плохие клетки и артефакты, задача станет проще. Если оставить их, задача станет труднее, но зато ближе к реальной практике.

Почему это важно:

- в реальной работе анализатора изображение редко бывает идеально чистым;
- если модель хороша только на "красивых" данных, то пользы от нее мало;
- поэтому нужно отдельно проверить и чистый режим, и реалистичный режим.

Как гипотеза проверялась:

1. Было создано 4 версии корпуса:
   - `full_realistic`;
   - `clean_cell`;
   - `strict_morphology`;
   - `mask_ready`.
2. Основное сравнение шума проводилось между:
   - `clean_cell`;
   - `strict_morphology`;
   - `full_realistic`.
3. Для каждой версии корпуса были обучены одни и те же признаки и одни и те же модели.
4. Затем в `A8` сравнивались итоговые таблицы:
   - насколько падает `PR-AUC`;
   - насколько меняется `recall blast`;
   - насколько снижается устойчивость модели.

Какие модели использовались для проверки:

- весь baseline-блок `A6`, то есть все 5 признаковых моделей;
- в итоговой абляции режимов корпуса сравнение фиксировалось на лучшей boosting-конфигурации.

Что будет считаться подтверждением гипотезы:

- если на более чистых корпусах качество выше;
- если на `full_realistic` качество немного падает, но не разрушается полностью;
- если модель остается practically useful даже при наличии артефактов.

### Гипотеза 4. Object-level оценки можно превратить в полезную оценку изображения и пациента

Смысл гипотезы:

даже если модель принимает решение по одной клетке, ее оценки можно объединить и понять:

- есть ли на изображении хотя бы один подозрительный объект;
- какие изображения стоит проверить в первую очередь;
- какого пациента стоит поднять вверх в списке риска.

Это важный шаг к задаче автоматического анализатора.

Как гипотеза проверялась:

1. Сначала выбиралась лучшая object-level baseline-модель для `full_realistic`.
2. Она выдавала score для каждого объекта.
3. Потом эти score агрегировались на уровень изображения и пациента несколькими способами:
   - `max_object_score`;
   - `mean_top3_scores`;
   - `count_objects_above_threshold`;
   - `soft_noisy_or`.
4. Для image-level и patient-level считались отдельные метрики:
   - `PR-AUC`;
   - `ROC-AUC`;
   - `recall`;
   - `top-k recall`.

Какие модели участвовали в проверке:

- напрямую не все модели, а одна лучшая object-level baseline-модель;
- однако сама эта лучшая модель выбиралась среди всех признаковых baseline-конфигураций блока `A6`.

Что будет считаться подтверждением гипотезы:

- если один или несколько способов агрегирования дадут высокие image-level метрики;
- если по ранжированию можно поднимать изображения и пациентов с бластами в верх списка;
- если object-level модель окажется полезной не только на одной клетке, но и как блок более крупной системы.

### Гипотеза 5. Легкие deep-модели можно честно сравнить с handcrafted-подходом на тех же данных

Смысл гипотезы:

часто считается, что нейросети автоматически лучше "ручных" признаков. Эксперимент проверяет, так ли это на данном конкретном корпусе.

Важно:

- сравнение должно быть честным;
- одинаковые пациенты должны быть в train/val/test;
- нельзя давать одной модели лучшие данные, чем другой.

Как гипотеза проверялась:

1. Для deep-блока был собран специальный `modeling_workset`.
2. Были обучены 3 модели:
   - `cnn_bbox` — CNN на `bbox_crop`;
   - `cnn_mask` — CNN на `mask_crop`;
   - `cnn_late_fusion` — CNN + табличные признаки.
3. Лучший handcrafted baseline из `A6` сравнивался с deep-моделями по тем же test-метрикам:
   - `PR-AUC`;
   - `ROC-AUC`;
   - `balanced accuracy`;
   - `macro-F1`;
   - `recall blast`;
   - `specificity`;
   - `MCC`;
   - `Brier score`;
   - `ECE`.

Почему отдельно есть `late fusion`:

- это очень важная промежуточная идея;
- она проверяет, что будет, если не противопоставлять нейросеть и handcrafted-признаки, а объединить их.

Что будет считаться подтверждением гипотезы:

- если deep-модели смогут догнать или превзойти лучший handcrafted baseline;
- или, наоборот, если окажется, что на текущем корпусе интерпретируемые признаки уже дают очень сильный результат и легкая CNN их не превосходит.

### Что в итоге проверялось на самом деле

Если свести все к одной схеме, то эксперимент проверял три больших вопроса:

1. Полезны ли вообще объяснимые признаки для распознавания бластов?
2. Какие именно условия помогают сильнее всего:
   - маска;
   - очистка корпуса;
   - определенная группа признаков;
   - определенная модель?
3. Можно ли из хорошей object-level модели сделать practically useful систему для image-level и patient-level анализа?

Именно поэтому в эксперименте использовались сразу несколько семейств моделей:

- простые линейные модели;
- нелинейные модели на признаках;
- ансамбли деревьев;
- легкие CNN;
- гибридная late-fusion модель.

Это позволило проверить гипотезы не "на словах", а в серии честных количественных сравнений.

## 3. На каких данных проводился эксперимент

### 3.1. Исходные данные

Использовались данные из:

- [Data/data](/Users/chqnb218718/Mephi/Dissertation/Experiments/Data/data)
- [Data/data_masks](/Users/chqnb218718/Mephi/Dissertation/Experiments/Data/data_masks)

Главный принцип был таким:

- в основной протокол брались только каталоги пациентов нормального формата;
- специальный каталог `part-1` специально не смешивался с основным patient-level корпусом.

### 3.2. Что получилось после аудита данных

По итогам этапа `A0` было установлено:

| Показатель | Значение |
|---|---:|
| Верхнеуровневые каталоги в `data` | 123 |
| Верхнеуровневые каталоги в `data_masks` | 123 |
| Каталоги пациентов в основном протоколе | 122 |
| Изображения в основном протоколе | 12 488 |
| Label-файлы в основном протоколе | 12 456 |
| Объекты в основном протоколе | 86 396 |
| Маски в основном протоколе | 86 360 |
| Изображения с хотя бы одной маской | 12 428 |
| Изображения без масок | 60 |
| Пациенты с бластами | 100 |
| Пациенты без бластов | 22 |

Подтверждение этих чисел находится в:

- [data_audit_runtime.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/data_audit_runtime.md)

### 3.3. Какие проблемы были найдены в данных

Во время аудита были найдены и учтены реальные ограничения:

- часть label-файлов использовала class id в формате `float`, например `7.0`;
- каталог `part-1` не был включен в основной patient-level протокол;
- `20` объектов имели некорректную геометрию bbox и были исключены из моделирования;
- не у всех объектов была маска.

### 3.4. Финальный modeling-корпус после геометрической очистки

После исключения `20` объектов с плохой геометрией осталось:

| Показатель | Значение |
|---|---:|
| Валидные объекты для моделирования | 86 376 |
| Объекты с корректно сопоставленной маской | 86 250 |
| Объекты без маски | 126 |
| Объекты с `mask_tight_crop` | 86 166 |
| Средняя ширина bbox | 131.41 px |
| Средняя высота bbox | 132.28 px |

Файлы этапа:

- [object_index.parquet](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/dataset_index/object_index.parquet)
- [bbox_roi_manifest.parquet](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/crops/bbox_roi_manifest.parquet)
- [roi_manifest.parquet](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/crops/roi_manifest.parquet)

## 4. Какие версии корпуса были созданы

Чтобы не смешивать "чистую" биологическую задачу и шумную реальную задачу, корпус был разбит на 4 режима.

| Подмножество | Что входит | Объектов | Пациентов | Изображений |
|---|---|---:|---:|---:|
| `full_realistic` | все объекты, включая плохие клетки и артефакты | 86 376 | 121 | 12 429 |
| `clean_cell` | без `Bad cells` и `artifacts` | 78 498 | 121 | 11 941 |
| `strict_morphology` | без `Bad cells`, `artifacts`, `gumpricht shadows` | 74 901 | 121 | 11 926 |
| `mask_ready` | только объекты с маской | 86 250 | 121 | 12 427 |

Это важно понимать так:

- `full_realistic` — это самый честный режим для реальной практики;
- `clean_cell` — более "чистая" задача;
- `strict_morphology` — еще более строгая постановка, где шум минимизирован;
- `mask_ready` — специальный режим для сравнения bbox и mask.

Подробные артефакты:

- [subset_summary.csv](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/tables/subset_summary.csv)
- [subset_class_distribution.csv](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/tables/subset_class_distribution.csv)
- [a1_subset_qc.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/a1_subset_qc.md)

## 5. Как данные были разделены на обучение и тест

Очень важная часть эксперимента: разделение делалось не по отдельным клеткам, а по пациентам.

Это означает:

- если пациент попал в train, то все его клетки и изображения были только в train;
- если пациент попал в test, то модель никогда раньше его не видела.

Это намного честнее, чем случайно перемешивать клетки.

### 5.1. Финальное разбиение

| Часть | Пациентов | Пациентов с объектами | Изображений | Объектов | Бластов |
|---|---:|---:|---:|---:|---:|
| `train` | 85 | 85 | 8 968 | 60 641 | 44 328 |
| `val` | 18 | 17 | 1 672 | 20 200 | 14 368 |
| `test` | 19 | 19 | 1 790 | 5 555 | 3 165 |

Замечание:

- в `val` один пациентский каталог есть в протоколе, но не содержит размеченных объектов;
- пересечений между `train`, `val`, `test` нет.

Файлы:

- [split_v1.json](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/splits/split_v1.json)
- [patient_summary.csv](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/splits/patient_summary.csv)
- [class_distribution_by_split.csv](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/tables/class_distribution_by_split.csv)
- [a2_split_qc.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/a2_split_qc.md)

## 6. Какие задачи классификации решались

Во всех задачах положительный класс определялся одинаково:

`target_binary = 1`, если `class_id == 7`, то есть объект является `blast`.

Были созданы 3 постановки.

| Задача | Что сравнивается | Объектов | Бластов | Небластов | Доля положительного класса |
|---|---|---:|---:|---:|---:|
| `A4.1` | `blast / non-blast` на `clean_cell` | 78 498 | 61 851 | 16 647 | 0.7879 |
| `A4.2` | `blast / non-blast` на `strict_morphology` | 74 901 | 61 851 | 13 050 | 0.8258 |
| `A4.3` | `blast / everything_else` на `full_realistic` | 86 376 | 61 851 | 24 525 | 0.7161 |

В `A4.3` самые важные hard-negative классы:

- `Bad cells`
- `Lymphocyte`
- `gumpricht shadows`

Это как раз делает задачу ближе к реальной работе автоматического анализатора.

Файлы:

- [task_a41_clean_binary.parquet](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/dataset_index/task_a41_clean_binary.parquet)
- [task_a42_strict_binary.parquet](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/dataset_index/task_a42_strict_binary.parquet)
- [task_a43_full_realistic_binary.parquet](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/dataset_index/task_a43_full_realistic_binary.parquet)
- [task_prevalence_summary.csv](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/tables/task_prevalence_summary.csv)

## 7. Как из клеток были сделаны изображения для анализа

Для каждого объекта было создано 3 вида ROI.

### 7.1. `bbox_crop`

Обычный прямоугольный фрагмент изображения по рамке детектора.

Просто говоря: вырезается кусочек вокруг клетки.

### 7.2. `mask_crop`

Берется тот же прямоугольный фрагмент, но пиксели вне маски зануляются.

Просто говоря: оставляем клетку, а все лишнее вокруг стараемся убрать.

### 7.3. `mask_tight_crop`

Это еще более плотная версия: вырезается минимальный прямоугольник вокруг самой маски.

Почему это важно:

- `bbox_crop` показывает, что модель видит, если мы не очень точно отделяем клетку от окружения;
- `mask_crop` показывает, что будет, если выделение клетки сделано аккуратно;
- сравнение этих режимов позволяет понять, насколько реально полезны маски.

Примеры ROI:

![Примеры ROI](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/figures/example_rois.png)

QC-файлы:

- [a3_bbox_qc.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/a3_bbox_qc.md)
- [a3_mask_qc.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/a3_mask_qc.md)

## 8. Какие признаки вычислялись и что они означают

Это один из самых важных разделов, потому что вся ценность эксперимента в том, что признаки можно объяснить человеку.

### 8.1. Морфометрические признаки

Это признаки формы и размера.

Примеры:

- площадь рамки;
- площадь маски;
- периметр;
- круглотность;
- эксцентриситет;
- заполненность bbox маской;
- solidity.

Понятный смысл:

- клетка может быть более круглой или более вытянутой;
- она может занимать почти весь bbox или только часть;
- контур может быть более плотным или более "неровным".

### 8.2. Цветовые признаки

Это числовое описание цвета клетки.

Использовались:

- средние значения по `RGB`;
- стандартные отклонения по `RGB`;
- квантили;
- признаки в пространствах `HSV` и `Lab`.

Понятный смысл:

- модель смотрит не просто на "какой цвет", а на распределение цвета;
- например, насколько цвет однородный, насколько он смещен в красную, зеленую или синюю сторону;
- `Lab` удобен тем, что он лучше разделяет яркость и цветовой оттенок.

### 8.3. Текстурные признаки

Это признаки "рисунка" внутри клетки.

Использовались:

- `GLCM` — анализ того, как часто рядом встречаются пиксели разной яркости;
- `LBP` — анализ локальных микрошаблонов текстуры;
- `Sobel` — оценка силы границ и перепадов;
- `Laplacian` — оценка резких изменений яркости;
- `Wavelet` — анализ структуры изображения на нескольких масштабах.

Понятный смысл:

- одни клетки выглядят более "гладкими";
- другие содержат более неоднородную структуру;
- у некоторых сильнее выражены мелкие детали и переходы.

### 8.4. Сколько признаков получилось

На полном корпусе были рассчитаны:

| Матрица признаков | Число строк |
|---|---:|
| `features_bbox.parquet` | 86 376 |
| `features_mask.parquet` | 86 250 |
| `features_combined.parquet` | 86 376 |

В объединенной матрице `combined` получилось `216` числовых признаков.

Из них:

- `109` признаков имели пропуски;
- `0` признаков были константными.

Почему были пропуски:

- часть признаков невозможна без маски;
- если маски нет, соответствующие mask-признаки не вычисляются.

Файлы:

- [features_bbox.parquet](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/features/features_bbox.parquet)
- [features_mask.parquet](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/features/features_mask.parquet)
- [features_combined.parquet](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/features/features_combined.parquet)
- [feature_dictionary.csv](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/features/feature_dictionary.csv)
- [feature_missingness.csv](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/tables/feature_missingness.csv)

## 9. Какие модели использовались

### 9.1. Признаковые baseline-модели

Было обучено `105` конфигураций:

- `3` задачи;
- `5` моделей;
- `7` групп признаков.

Использовались следующие модели.

#### Logistic Regression

Это простая линейная модель.

Она пытается провести границу между классами, используя взвешенную сумму признаков.

Плюс:

- хорошо интерпретируется.

Минус:

- не умеет хорошо описывать очень сложные нелинейные зависимости.

#### Linear SVM

Тоже линейная модель, но с другой логикой поиска разделяющей границы.

Плюс:

- часто хорошо работает на табличных признаках;
- устойчив при большом числе признаков.

#### RBF SVM

Это уже нелинейный вариант SVM.

В коде он реализован через приближение `Nystroem`, чтобы обучение не было слишком тяжелым.

Плюс:

- умеет строить более сложную границу между классами.

#### Random Forest

Это ансамбль деревьев решений.

Простое объяснение:

- строится много деревьев;
- каждое дерево принимает решение по-своему;
- потом они голосуют.

#### Gradient Boosting

В таблицах он идет под именем `xgboost`, но есть важная техническая оговорка.

В этой вычислительной среде настоящий `XGBoost` не загрузился из-за отсутствия `libomp.dylib`, поэтому фактически использовался fallback:

- `HistGradientBoostingClassifier`

Это тоже градиентный бустинг по деревьям, но не совсем тот же самый алгоритм, что "настоящий" XGBoost.

Это важно помнить при чтении результатов.

### 9.2. Группы признаков

Для каждой модели сравнивались:

- только морфометрия;
- только цвет;
- только текстура;
- морфометрия + цвет;
- морфометрия + текстура;
- цвет + текстура;
- полный набор признаков.

## 10. Какие метрики использовались и как их понимать

Ниже очень короткое объяснение каждой метрики простым языком.

### PR-AUC

Хороша, когда положительный класс важен.

Здесь это особенно важно, потому что нам нужно находить бласты.

Чем выше `PR-AUC`, тем лучше модель отделяет важный положительный класс от остальных.

### ROC-AUC

Показывает, насколько хорошо модель в целом отличает два класса.

### Balanced Accuracy

Обычная accuracy может обманывать на несбалансированных данных.

Balanced accuracy дает одинаковый вес обоим классам.

### Macro-F1

Это компромисс между точностью и полнотой, рассчитанный так, чтобы оба класса были важны.

### Recall для `blast`

Очень важная метрика.

Она отвечает на вопрос:

сколько настоящих бластов модель смогла найти.

### Specificity

Отвечает на вопрос:

сколько небластов модель правильно не приняла за бласт.

### Matthews Correlation Coefficient (MCC)

Одна из самых честных сводных метрик для бинарной классификации.

Чем ближе к `1`, тем лучше.

### Brier Score

Показывает, насколько хорошо вероятности модели соответствуют реальности.

Меньше — лучше.

### Expected Calibration Error (ECE)

Показывает, насколько уверенность модели совпадает с ее реальной точностью.

Меньше — лучше.

## 11. Главные object-level результаты

### 11.1. Лучшая модель в каждой задаче

| Задача | Лучшая конфигурация | Validation PR-AUC | Test PR-AUC | Recall blast | Specificity | MCC |
|---|---|---:|---:|---:|---:|---:|
| `A4.1 clean_cell` | `linear_svm + full` | 0.9760 | 0.9319 | 0.8961 | 0.7352 | 0.6452 |
| `A4.2 strict_morphology` | `linear_svm + color_texture` | 0.9848 | 0.9430 | 0.9305 | 0.6745 | 0.6409 |
| `A4.3 full_realistic` | `gradient boosting (labelled xgboost) + full` | 0.9143 | 0.9390 | 0.8847 | 0.7962 | 0.6859 |

Что это означает простыми словами:

- на самых "чистых" версиях корпуса очень хорошо работает линейный подход;
- на самой реалистичной задаче лучший результат дал полный набор признаков вместе с бустингом;
- даже в шумном режиме `full_realistic` модель сохраняет высокий `PR-AUC = 0.9390`.

### 11.2. Почему результат на `full_realistic` особенно важен

Именно эта задача ближе всего к реальному использованию, потому что здесь сохраняются:

- плохие клетки;
- артефакты;
- тени Гумпрехта;
- другие небластные объекты.

То есть модель должна не просто отличить "идеальный бласт" от "идеального небласта", а работать в реальном шумном окружении.

Подробные файлы:

- [object_level_metrics_baselines.csv](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/tables/object_level_metrics_baselines.csv)
- [baseline_model_leaderboard.csv](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/tables/baseline_model_leaderboard.csv)

### 11.3. Кривые качества и матрица ошибок

PR/ROC-кривые:

![PR/ROC кривые](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/figures/pr_curves_baselines.png)

![ROC кривые](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/figures/roc_curves_baselines.png)

Матрица ошибок лучшего baseline:

![Confusion matrix](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/figures/confusion_matrix_best_baseline.png)

## 12. Что показали абляции

Абляция — это специальный эксперимент, где мы убираем или меняем одну часть системы и смотрим, что изменится.

### 12.1. Роль маски

Наиболее показательная задача для практики — `A4.3 full_realistic`.

| Режим признаков | PR-AUC | ROC-AUC | Recall blast | Specificity |
|---|---:|---:|---:|---:|
| `bbox_only` | 0.9148 | 0.8991 | 0.7938 | 0.8191 |
| `mask_only` | 0.9175 | 0.8984 | 0.8735 | 0.7401 |
| `bbox_plus_mask` | 0.9393 | 0.9259 | 0.9130 | 0.7523 |

Вывод:

- маска сама по себе уже полезна;
- лучше всего работает объединение `bbox + mask`;
- значит, информация и из общего контекста клетки, и из аккуратно выделенной области важна одновременно.

### 12.2. Роль шумовых классов

Сравнение лучших полных режимов:

| Задача | PR-AUC | ROC-AUC | Recall blast |
|---|---:|---:|---:|
| `clean_cell` | 0.9518 | 0.9335 | 0.9741 |
| `strict_morphology` | 0.9598 | 0.9287 | 0.9596 |
| `full_realistic` | 0.9390 | 0.9260 | 0.8847 |

Главный вывод:

- когда в данных остаются реальные шумовые классы, задача действительно становится сложнее;
- но падение качества не катастрофическое;
- значит, признаки остаются полезными и в более реалистичном режиме.

### 12.3. Class weights и отбор признаков

На `full_realistic`:

| Режим | PR-AUC | Recall blast | MCC |
|---|---:|---:|---:|
| `combined_with_class_weight` | 0.9390 | 0.8847 | 0.6859 |
| `combined_without_class_weight` | 0.9390 | 0.8847 | 0.6859 |
| `combined_top30_features` | 0.9362 | 0.8139 | 0.6699 |

Интерпретация:

- в этом запуске class weights не дали видимого изменения;
- компактный набор из `30` лучших признаков почти сохраняет качество по `PR-AUC`, но заметно проседает по `recall blast`;
- это важный практический результат: сильное сжатие модели возможно, но часть чувствительности теряется.

Файлы:

- [ablation_mask_usage.csv](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/tables/ablation_mask_usage.csv)
- [ablation_dataset_regimes.csv](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/tables/ablation_dataset_regimes.csv)
- [ablation_summary.png](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/figures/ablation_summary.png)

## 13. Какие признаки оказались самыми важными

Для лучшей модели на `full_realistic` был проведен анализ важности признаков.

Самые стабильные признаки:

| Признак | Частота попадания в top-20 | Что это примерно значит |
|---|---:|---|
| `mask_lab_a_std` | 3 | насколько по маске меняется цвет по оси green-red |
| `bbox_lab_a_std` | 3 | тот же цветовой разброс, но в bbox |
| `bbox_lab_b_q75` | 3 | верхний квартиль цвета по оси blue-yellow |
| `bbox_hsv_h_mean` | 3 | средний цветовой тон |
| `mask_texture_glcm_correlation_std` | 3 | насколько неоднородна текстурная корреляция |
| `mask_morph_solidity` | 3 | насколько форма клетки плотная и цельная |

Очень важный общий смысл:

- признаки цвета оказались не менее важны, чем признаки формы;
- признаки текстуры действительно входят в число устойчиво полезных;
- это хорошо согласуется с исходной научной мотивацией эксперимента.

Фигуры:

![Стабильность признаков](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/figures/feature_importance_stability.png)

![Сводка важности признаков](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/figures/shap_summary_top_model.png)

Файлы:

- [top_features_stability.csv](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/tables/top_features_stability.csv)
- [feature_interpretation_notes.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/feature_interpretation_notes.md)

## 14. Что получилось на уровне изображения и пациента

Это очень важный мост между "классификацией одной клетки" и реальной задачей анализатора.

### 14.1. Image-level результаты

На тесте было:

- `1790` изображений;
- `887` изображений содержали хотя бы один бласт.

Сравнивались 4 способа агрегирования object-level score:

- `max_object_score`
- `mean_top3_scores`
- `count_objects_above_threshold`
- `soft_noisy_or`

Лучшая image-level схема по `PR-AUC`:

| Метод | PR-AUC | ROC-AUC | Recall blast | Specificity | Recall at FPR 10% |
|---|---:|---:|---:|---:|---:|
| `soft_noisy_or` | 0.9743 | 0.9727 | 0.9076 | 0.9003 | 0.9076 |

Простое объяснение `soft noisy-or`:

- если хотя бы один объект на изображении очень похож на бласт, то изображение становится подозрительным;
- при этом метод не реагирует слишком грубо на слабые шумовые сигналы.

Вывод:

- object-level модель действительно можно использовать как основу для отбора подозрительных изображений;
- это сильный аргумент в пользу практической полезности эксперимента.

### 14.2. Patient-level результаты

На patient-level тест состоял из:

- `19` пациентов;
- `15` пациентов с бластами.

Лучшая patient-level схема по `PR-AUC`:

| Метод | PR-AUC | ROC-AUC | Recall blast | Specificity |
|---|---:|---:|---:|---:|
| `count_objects_above_threshold` | 0.9860 | 0.9333 | 0.6667 | 1.0000 |

Здесь важно быть осторожным:

- patient-level тест маленький (`19` пациентов), поэтому числа чувствительны к каждому случаю;
- тем не менее, даже на таком небольшом тесте видно, что пациент с бластами поднимается вверх в ранжированном списке достаточно хорошо.

Фигуры:

![Image ranking recall](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/figures/image_ranking_recall_curve.png)

![Patient ranking curve](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/figures/patient_ranking_curve.png)

Файлы:

- [image_level_metrics.csv](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/tables/image_level_metrics.csv)
- [patient_level_ranking_metrics.csv](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/tables/patient_level_ranking_metrics.csv)

## 15. Deep-baselines: что было сделано и что получилось

### 15.1. Какие deep-модели сравнивались

Были обучены 3 легкие модели:

1. `cnn_bbox`
   Модель видит только `bbox_crop`.

2. `cnn_mask`
   Модель видит только `mask_crop`.

3. `cnn_late_fusion`
   Модель видит изображение и одновременно табличные признаки.

Все они обучались на одном и том же `modeling_workset`.

### 15.2. Объем данных для deep-этапа

Для deep-блока использовался рабочий набор:

- `30 501` mask-ready объектов для `full_realistic`;
- те же объекты использовались для late fusion.

### 15.3. Графики обучения

Графики:

![Графики обучения CNN](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/figures/deep_training_curves.png)

Краткие факты из истории обучения:

| Модель | Эпох | Лучшая эпоха | Лучшая val PR-AUC |
|---|---:|---:|---:|
| `cnn_bbox` | 3 | 1 | 0.7742 |
| `cnn_mask` | 3 | 1 | 0.7862 |
| `cnn_late_fusion` | 5 | 3 | 0.8932 |

Что это значит:

- чисто визуальные CNN на bbox и mask быстро выходили на максимум и дальше не росли;
- late fusion учился дольше и заметно лучше;
- добавление табличных признаков действительно помогло deep-подходу.

### 15.4. Сравнение handcrafted и deep

| Модель | Validation PR-AUC | Test PR-AUC | Test ROC-AUC | Recall blast | MCC |
|---|---:|---:|---:|---:|---:|
| `handcrafted_best_full_realistic` | 0.9143 | 0.9390 | 0.9260 | 0.8847 | 0.6859 |
| `cnn_bbox` | 0.7742 | 0.7491 | 0.7203 | 0.5019 | 0.2426 |
| `cnn_mask` | 0.7862 | 0.7119 | 0.6311 | 0.4168 | 0.1634 |
| `cnn_late_fusion` | 0.8932 | 0.8931 | 0.8690 | 0.8612 | 0.5485 |

Главный вывод:

- лучший handcrafted baseline оказался сильнее всех deep-baselines;
- лучшая deep-модель — это `cnn_late_fusion`;
- late fusion заметно лучше чисто визуальных CNN, но все равно не превзошел лучший признаковый подход.

Это очень важный научный результат:

на данном корпусе, в данной постановке и при данной вычислительной сложности интерпретируемые признаки не просто "не хуже", а реально сильнее легких CNN.

Файлы:

- [deep_vs_handcrafted.csv](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/tables/deep_vs_handcrafted.csv)
- [cnn_training_history.json](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/logs/cnn_training_history.json)
- [a9_deep_qc.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/a9_deep_qc.md)

## 16. Главные выводы простыми словами

### Вывод 1

Да, интерпретируемые признаки формы, цвета и текстуры действительно полезны.

Это подтверждается высокими значениями `PR-AUC` и устойчивыми результатами на patient-level split.

### Вывод 2

Наиболее сильный и практический результат был получен на задаче `full_realistic`, где сохранены шумовые классы.

Это особенно ценно, потому что такая постановка ближе к реальным медицинским данным.

### Вывод 3

Маски помогают.

Связка `bbox + mask` почти везде лучше, чем `bbox` или `mask` по отдельности.

### Вывод 4

Текстурные признаки не оказались "лишними".

Они входят в число устойчиво полезных признаков вместе с цветовыми и морфометрическими характеристиками.

### Вывод 5

Object-level модель можно достаточно успешно поднимать до image-level и patient-level оценки.

Это делает эксперимент полезным не только как задачу "классификации клетки", но и как шаг к автоматическому анализатору.

### Вывод 6

Легкие CNN на этом корпусе не превзошли лучший handcrafted baseline.

Это означает, что простой и объяснимый подход здесь не проигрывает "нейросетевой моде", а наоборот показывает очень сильный результат.

## 17. Что эксперимент НЕ доказывает

Важно честно зафиксировать ограничения.

Этот эксперимент не доказывает:

- различие `lymphoblast / myeloblast`;
- переносимость на другую лабораторию;
- переносимость на другой сканер или другую окраску;
- полноценную slide-level диагностику;
- превосходство настоящего `XGBoost`, потому что в данной среде фактически использовался fallback `HistGradientBoosting`.

Это не недостаток отчета, а честная граница применимости результатов.

## 18. Технические и научные ограничения

### 18.1. Техническое ограничение среды

В `.venv` не загрузился реальный `xgboost`, потому что отсутствует `libomp.dylib`.

Поэтому в тех строках, где модель подписана как `xgboost`, фактически использовался:

- `HistGradientBoostingClassifier`

Это честно отражено в таблицах и QC-отчетах.

### 18.2. Ограничение данных

- нет разделения бластов на подтипы;
- patient-level test относительно небольшой;
- часть объектов без маски;
- один patient-каталог из основного протокола не содержит размеченных объектов.

## 19. Где лежат главные артефакты

### Основные таблицы

- [object_level_metrics_baselines.csv](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/tables/object_level_metrics_baselines.csv)
- [baseline_model_leaderboard.csv](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/tables/baseline_model_leaderboard.csv)
- [feature_group_ablation.csv](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/tables/feature_group_ablation.csv)
- [ablation_mask_usage.csv](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/tables/ablation_mask_usage.csv)
- [ablation_dataset_regimes.csv](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/tables/ablation_dataset_regimes.csv)
- [image_level_metrics.csv](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/tables/image_level_metrics.csv)
- [patient_level_ranking_metrics.csv](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/tables/patient_level_ranking_metrics.csv)
- [deep_vs_handcrafted.csv](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/tables/deep_vs_handcrafted.csv)

### Основные рисунки

- [pipeline_overview.png](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/figures/pipeline_overview.png)
- [example_rois.png](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/figures/example_rois.png)
- [mask_coverage.png](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/figures/mask_coverage.png)
- [pr_curves_baselines.png](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/figures/pr_curves_baselines.png)
- [roc_curves_baselines.png](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/figures/roc_curves_baselines.png)
- [confusion_matrix_best_baseline.png](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/figures/confusion_matrix_best_baseline.png)
- [feature_importance_stability.png](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/figures/feature_importance_stability.png)
- [shap_summary_top_model.png](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/figures/shap_summary_top_model.png)
- [ablation_summary.png](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/figures/ablation_summary.png)
- [image_ranking_recall_curve.png](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/figures/image_ranking_recall_curve.png)
- [patient_ranking_curve.png](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/figures/patient_ranking_curve.png)
- [deep_training_curves.png](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/figures/deep_training_curves.png)

### QC-отчеты

- [data_audit_runtime.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/data_audit_runtime.md)
- [a1_subset_qc.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/a1_subset_qc.md)
- [a2_split_qc.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/a2_split_qc.md)
- [a3_bbox_qc.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/a3_bbox_qc.md)
- [a3_mask_qc.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/a3_mask_qc.md)
- [feature_extraction_qc.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/feature_extraction_qc.md)
- [a6_baseline_qc.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/a6_baseline_qc.md)
- [a10_a11_qc.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/a10_a11_qc.md)
- [a9_deep_qc.md](/Users/chqnb218718/Mephi/Dissertation/Experiments/E01_textural_features_blast_pipeline/outputs/reports/a9_deep_qc.md)

## 20. Короткий итог в одном абзаце

Эксперимент показал, что на текущем корпусе данных интерпретируемые морфометрические, цветовые и текстурные признаки действительно позволяют хорошо различать `blast / non-blast` на patient-level split, причем особенно ценно то, что сильный результат сохраняется даже в шумном `full_realistic` режиме. Маски полезны, объединение `bbox + mask` лучше отдельных вариантов, а объектные оценки успешно поднимаются до image-level и patient-level. Легкие CNN оказались слабее лучшего handcrafted baseline, а лучшая deep-модель — это late fusion изображения и табличных признаков. В результате эксперимент уже сейчас дает содержательный, воспроизводимый и понятный для диссертации набор выводов.
