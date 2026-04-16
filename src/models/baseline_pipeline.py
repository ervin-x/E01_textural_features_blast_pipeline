from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


KEY_COLUMNS = [
    "patient_id",
    "image_id",
    "object_id_within_image",
    "class_id",
    "class_name",
    "has_mask",
]


META_COLUMNS = KEY_COLUMNS + [
    "target_binary",
    "split_name",
    "geometry_valid",
    "geometry_issue",
    "mask_missing",
]


FEATURE_GROUPS = {
    "morphology": ("morphology",),
    "color": ("color",),
    "texture": ("texture",),
    "morphology_color": ("morphology", "color"),
    "morphology_texture": ("morphology", "texture"),
    "color_texture": ("color", "texture"),
    "full": ("morphology", "color", "texture"),
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    implementation: str
    uses_scaling: bool


def column_family(column_name: str) -> str | None:
    if column_name in META_COLUMNS:
        return None
    normalized = column_name
    if normalized.startswith("bbox_"):
        normalized = normalized[len("bbox_") :]
    elif normalized.startswith("mask_"):
        normalized = normalized[len("mask_") :]

    if normalized.startswith("morph_"):
        return "morphology"
    if normalized.startswith(("color_", "hsv_", "lab_")):
        return "color"
    if normalized.startswith(("texture_", "wavelet_")):
        return "texture"
    return None


def feature_columns_by_group(feature_df: pd.DataFrame) -> dict[str, list[str]]:
    family_to_columns: dict[str, list[str]] = {"morphology": [], "color": [], "texture": []}
    for column in feature_df.columns:
        family = column_family(column)
        if family is None:
            continue
        family_to_columns[family].append(column)

    groups: dict[str, list[str]] = {}
    for group_name, families in FEATURE_GROUPS.items():
        columns: list[str] = []
        for family in families:
            columns.extend(family_to_columns[family])
        groups[group_name] = sorted(columns)
    return groups


def merge_task_with_features(task_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    merged = task_df.merge(
        feature_df,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) == 0:
        raise ValueError("Task-feature merge produced zero rows.")
    return merged


def build_model(model_name: str, scale_pos_weight: float, use_class_weight: bool = True) -> tuple[Pipeline, ModelSpec]:
    common_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    class_weight = "balanced" if use_class_weight else None

    if model_name == "logistic_regression":
        pipeline = Pipeline(
            steps=[
                *common_steps,
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight=class_weight,
                        solver="lbfgs",
                        n_jobs=None,
                        random_state=42,
                    ),
                ),
            ]
        )
        return pipeline, ModelSpec("logistic_regression", "exact", True)

    if model_name == "linear_svm":
        base = LinearSVC(
            class_weight=class_weight,
            max_iter=5000,
            dual="auto",
            random_state=42,
        )
        pipeline = Pipeline(
            steps=[
                *common_steps,
                ("scaler", StandardScaler()),
                ("classifier", CalibratedClassifierCV(base, cv=3, method="sigmoid")),
            ]
        )
        return pipeline, ModelSpec("linear_svm", "exact", True)

    if model_name == "rbf_svm":
        base = LinearSVC(
            class_weight=class_weight,
            max_iter=5000,
            dual="auto",
            random_state=42,
        )
        pipeline = Pipeline(
            steps=[
                *common_steps,
                ("scaler", StandardScaler()),
                ("rbf_map", Nystroem(kernel="rbf", gamma=0.5, n_components=128, random_state=42)),
                ("classifier", CalibratedClassifierCV(base, cv=3, method="sigmoid")),
            ]
        )
        return pipeline, ModelSpec("rbf_svm", "nystroem_approximation", True)

    if model_name == "random_forest":
        pipeline = Pipeline(
            steps=[
                *common_steps,
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample" if use_class_weight else None,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        )
        return pipeline, ModelSpec("random_forest", "exact", False)

    if model_name == "xgboost":
        if XGBOOST_AVAILABLE:
            pipeline = Pipeline(
                steps=[
                    *common_steps,
                    (
                        "classifier",
                        XGBClassifier(
                            n_estimators=200,
                            max_depth=6,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            reg_lambda=1.0,
                            min_child_weight=1.0,
                            objective="binary:logistic",
                            eval_metric="logloss",
                            tree_method="hist",
                            n_jobs=8,
                            random_state=42,
                            scale_pos_weight=scale_pos_weight if use_class_weight else 1.0,
                        ),
                    ),
                ]
            )
            return pipeline, ModelSpec("xgboost", "exact", False)

        pipeline = Pipeline(
            steps=[
                *common_steps,
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_depth=6,
                        learning_rate=0.05,
                        max_iter=200,
                        random_state=42,
                    ),
                ),
            ]
        )
        return pipeline, ModelSpec("xgboost", "hist_gradient_boosting_fallback", False)

    raise ValueError(f"Unsupported model: {model_name}")


def fit_predict_probability(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
) -> np.ndarray:
    pipeline.fit(X_train, y_train)
    if hasattr(pipeline, "predict_proba"):
        return np.asarray(pipeline.predict_proba(X_eval)[:, 1], dtype=float)

    decision = np.asarray(pipeline.decision_function(X_eval), dtype=float)
    return 1.0 / (1.0 + np.exp(-decision))
