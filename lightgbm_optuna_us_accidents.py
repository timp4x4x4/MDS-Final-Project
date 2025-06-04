# -*- coding: utf-8 -*-
"""
US-Accidents Severity Classification with LightGBM + Optuna
===========================================================
Fixed version (2025‑06‑04)
-------------------------
*  修正 `OneHotEncoder` 在 **scikit‑learn ≥ 1.4** 已移除 `sparse=` 參數而改為
   `sparse_output=` 的問題。現在採用 try/except 判斷版本，自動選擇適用的
   參數名稱，確保與新舊 scikit‑learn 皆相容。
*  其餘流程不變：資料讀取、前處理、SMOTE（可選）、Optuna 調參、最佳
   模型輸出。
"""

import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import lightgbm as lgb
import optuna

# Optional: SMOTE for class balancing
try:
    from imblearn.over_sampling import SMOTE
except ImportError:  # keep script runnable even without imbalanced-learn
    SMOTE = None

# -----------------------------
# Utility functions
# -----------------------------

def macro_f1(y_true, y_pred) -> float:
    """Convenience wrapper: macro-averaged F1-score."""
    return f1_score(y_true, y_pred, average="macro")


def load_data(csv_path: Path) -> pd.DataFrame:
    """Read the US Accidents CSV and return a cleaned DataFrame."""
    df = pd.read_csv(csv_path)

    # Basic cleaning – drop obviously non-predictive IDs
    df = df.drop(columns=[
        "ID", "Source", "Description", "Number", "Street", "City", "County",
        "State", "Country", "Timezone", "Airport_Code", "Weather_Timestamp",
        "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight",
    ], errors="ignore")

    # Parse Start_Time to datetime and extract components
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df["Start_Year"] = df["Start_Time"].dt.year
    df["Start_Month"] = df["Start_Time"].dt.month
    df["Start_Hour"] = df["Start_Time"].dt.hour
    df = df.drop(columns=["Start_Time"])

    # Drop rows with NaN Severity (shouldn't exist)
    df = df.dropna(subset=["Severity"])

    # Fill remaining numeric NaNs with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Cap outliers
    if "Wind_Speed(mph)" in df.columns:
        df.loc[df["Wind_Speed(mph)"] > 90, "Wind_Speed(mph)"] = 90
    if "Visibility(mi)" in df.columns:
        df.loc[df["Visibility(mi)"] > 20, "Visibility(mi)"] = 20

    return df


def prepare_features(df: pd.DataFrame):
    """Split df into X (features) and y (labels), returning train/valid/test sets."""
    y = df["Severity"].astype(int)
    X = df.drop(columns=["Severity"])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


# -----------------------------
# Pipeline builder (handles sklearn version)
# -----------------------------

def build_pipeline(categorical_cols):
    """Create a sklearn Pipeline that handles categorical encodings + LightGBM."""

    # Handle OneHotEncoder API change (sparse -> sparse_output)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        # Older versions (<1.2) fall back to 'sparse'
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    preprocessor = ColumnTransformer([
        ("cat", ohe, categorical_cols),
    ], remainder="passthrough")

    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=4,
        n_jobs=-1,
        boosting_type="gbdt",
        verbose=-1,
        metric="None",  # custom eval metric used
    )

    return Pipeline(steps=[("pre", preprocessor), ("clf", clf)])


# -----------------------------
# Optuna objective
# -----------------------------

def objective(trial, X_train, X_valid, y_train, y_valid, categorical_cols):
    pipe = build_pipeline(categorical_cols)

    params = {
        "clf__learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "clf__num_leaves": trial.suggest_int("num_leaves", 31, 512, log=True),
        "clf__max_depth": trial.suggest_int("max_depth", -1, 16),
        "clf__n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "clf__min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "clf__subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "clf__colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "clf__reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "clf__reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "clf__is_unbalance": True,
    }

    pipe.set_params(**params)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_valid)
    return macro_f1(y_valid, preds)


# -----------------------------
# Main routine
# -----------------------------

def train_and_evaluate(csv_path: Path, smote: bool):
    df = load_data(csv_path)
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if smote:
        if SMOTE is None:
            raise RuntimeError("imbalanced-learn not installed; install or omit --smote flag.")
        X_num = df.drop(columns=["Severity"])
        y_num = df["Severity"].astype(int)
        sampler = SMOTE(random_state=42)
        X_res, y_res = sampler.fit_resample(X_num, y_num)
        df = pd.concat([X_res, y_res], axis=1)

    X_train, X_valid, X_test, y_train, y_valid, y_test = prepare_features(df)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, X_valid, y_train, y_valid, categorical_cols),
        n_trials=50,
        timeout=3600,
    )

    print("Best macro‑F1:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print("  ", k, "=", v)

    # Train final model on train+valid using best params
    best_pipe = build_pipeline(categorical_cols)
    best_pipe.set_params(**{f"clf__{k}": v for k, v in study.best_params.items()})
    best_pipe.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))

    y_pred = best_pipe.predict(X_test)
    print("\n=== Test set performance ===")
    print("Macro‑F1:", macro_f1(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    from joblib import dump
    out_path = Path("best_lightgbm_optuna.pkl")
    dump(best_pipe, out_path)
    print(f"Model saved to {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGBM + Optuna for US Accidents")
    parser.add_argument("--csv", type=Path, required=True, help="Path to US_Accidents CSV file")
    parser.add_argument("--smote", action="store_true", help="Enable SMOTE oversampling")
    args = parser.parse_args()

    train_and_evaluate(args.csv, args.smote)
