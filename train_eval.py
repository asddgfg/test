# train_eval.py
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from splitter import train_dev_test_split, rolling_window_splits, expanding_window_splits
from models import get_regression_models, get_classification_models


PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
MODEL_DIR = "models_saved"


DATASET_CONFIG = {
    "dataset_return.csv": {
        "target": "y_return",
        "task": "regression",
        "purge": 1,
        "embargo": 1,
    },
    "dataset_direction.csv": {
        "target": "y_direction",
        "task": "classification",
        "purge": 1,
        "embargo": 1,
    },
    "dataset_volatility.csv": {
        "target": "y_vol",
        "task": "regression",
        "purge": 5,
        "embargo": 5,
    },
    "dataset_regime.csv": {
        "target": "y_regime",
        "task": "classification",
        "purge": 20,
        "embargo": 20,
    },
}


def ensure_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def get_expected_model_names(task: str) -> List[str]:
    if task == "regression":
        return list(get_regression_models().keys())
    if task == "classification":
        return list(get_classification_models().keys())
    raise ValueError(f"Unknown task type: {task}")


def get_model_path(dataset_name: str, model_name: str) -> str:
    safe_dataset = dataset_name.replace(".csv", "")
    return os.path.join(MODEL_DIR, f"{safe_dataset}__{model_name}.pkl")


def model_exists(dataset_name: str, model_name: str) -> bool:
    path = get_model_path(dataset_name, model_name)
    return os.path.exists(path) and os.path.getsize(path) > 0


def save_model(model, dataset_name: str, model_name: str) -> str:
    path = get_model_path(dataset_name, model_name)
    joblib.dump(model, path)
    return path


def load_model(dataset_name: str, model_name: str):
    path = get_model_path(dataset_name, model_name)
    return joblib.load(path)


def load_dataset(file_name: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing processed dataset: {path}")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def get_feature_target_split(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    feature_cols = [col for col in df.columns if col not in ["Date", target_col]]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    direction_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))

    return {
        "rmse": rmse,
        "mae": mae,
        "oos_r2": r2,
        "directional_accuracy": direction_acc,
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["roc_auc"] = np.nan

    return out


def evaluate_one_fold(model, X_train, y_train, X_valid, y_valid, task: str) -> Dict[str, float]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    if task == "regression":
        return regression_metrics(y_valid.values, y_pred)

    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_valid)[:, 1]

    return classification_metrics(y_valid.values, y_pred, y_prob)


def summarize_fold_metrics(records: List[Dict[str, float]]) -> Dict[str, float]:
    if len(records) == 0:
        return {}

    df = pd.DataFrame(records)
    summary = {}

    metric_cols = [col for col in df.columns if col != "fold"]
    for col in metric_cols:
        summary[f"{col}_cv_mean"] = float(df[col].mean())
        summary[f"{col}_cv_std"] = float(df[col].std())

    return summary


def evaluate_on_test(model, X_test, y_test, task: str) -> Dict[str, float]:
    y_test_pred = model.predict(X_test)

    if task == "regression":
        return regression_metrics(y_test.values, y_test_pred)

    y_test_prob = None
    if hasattr(model, "predict_proba"):
        y_test_prob = model.predict_proba(X_test)[:, 1]

    return classification_metrics(y_test.values, y_test_pred, y_test_prob)


def run_cv_for_dataset(
    file_name: str,
    split_mode: str = "expanding",
    test_size: float = 0.2,
    train_size: int = 504,
    min_train_size: int = 504,
    valid_size: int = 63,
    step_size: int = 21,
) -> pd.DataFrame:
    ensure_dirs()

    config = DATASET_CONFIG[file_name]
    target_col = config["target"]
    task = config["task"]
    purge = config["purge"]
    embargo = config["embargo"]

    df = load_dataset(file_name)
    dev_df, test_df = train_dev_test_split(df, test_size=test_size)

    X_dev, y_dev = get_feature_target_split(dev_df, target_col)
    X_test, y_test = get_feature_target_split(test_df, target_col)

    if task == "regression":
        models = get_regression_models()
    elif task == "classification":
        models = get_classification_models()
    else:
        raise ValueError(f"Unknown task type: {task}")

    if split_mode == "rolling":
        folds = list(
            rolling_window_splits(
                n_samples=len(dev_df),
                train_size=train_size,
                valid_size=valid_size,
                step_size=step_size,
                purge=purge,
                embargo=embargo,
            )
        )
    elif split_mode == "expanding":
        folds = list(
            expanding_window_splits(
                n_samples=len(dev_df),
                min_train_size=min_train_size,
                valid_size=valid_size,
                step_size=step_size,
                purge=purge,
                embargo=embargo,
            )
        )
    else:
        raise ValueError("split_mode must be 'rolling' or 'expanding'.")

    all_results = []

    for model_name, fresh_model in models.items():
        model_path = get_model_path(file_name, model_name)

        if model_exists(file_name, model_name):
            print(f"Loading existing model: {file_name} / {model_name}")
            model = load_model(file_name, model_name)

            test_metrics = evaluate_on_test(model, X_test, y_test, task)

            row = {
                "dataset": file_name,
                "task": task,
                "model": model_name,
                "model_path": model_path,
                "trained_this_run": False,
                "n_dev": len(dev_df),
                "n_test": len(test_df),
                "n_folds": len(folds),
            }
            for k, v in test_metrics.items():
                row[f"test_{k}"] = v

            all_results.append(row)
            continue

        print(f"Training missing model: {file_name} / {model_name}")
        model = fresh_model
        fold_records = []

        for i, fold in enumerate(folds, start=1):
            X_train = X_dev.iloc[fold.train_idx]
            y_train = y_dev.iloc[fold.train_idx]
            X_valid = X_dev.iloc[fold.valid_idx]
            y_valid = y_dev.iloc[fold.valid_idx]

            metrics = evaluate_one_fold(model, X_train, y_train, X_valid, y_valid, task)
            metrics["fold"] = i
            fold_records.append(metrics)

        cv_summary = summarize_fold_metrics(fold_records)

        model.fit(X_dev, y_dev)
        save_model(model, file_name, model_name)

        test_metrics = evaluate_on_test(model, X_test, y_test, task)

        row = {
            "dataset": file_name,
            "task": task,
            "model": model_name,
            "model_path": model_path,
            "trained_this_run": True,
            "n_dev": len(dev_df),
            "n_test": len(test_df),
            "n_folds": len(folds),
            **cv_summary,
        }

        for k, v in test_metrics.items():
            row[f"test_{k}"] = v

        all_results.append(row)

    return pd.DataFrame(all_results)


def run_all_datasets(split_mode: str = "expanding") -> pd.DataFrame:
    ensure_dirs()

    result_frames = []
    for file_name in DATASET_CONFIG:
        print(f"Running dataset: {file_name}")
        result_df = run_cv_for_dataset(file_name, split_mode=split_mode)
        result_frames.append(result_df)

    final_df = pd.concat(result_frames, axis=0, ignore_index=True)
    out_path = os.path.join(RESULTS_DIR, f"model_comparison_{split_mode}.csv")
    final_df.to_csv(out_path, index=False)

    print(f"Saved results to {out_path}")
    return final_df


if __name__ == "__main__":
    df = run_all_datasets(split_mode="expanding")
    print(df)