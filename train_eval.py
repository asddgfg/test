# train_eval.py
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from bootstrap_utils import (
    BootstrapSample,
    iter_bootstrap_samples,
    suggest_block_length,
)
from models import get_classification_models, get_regression_models
from splitter import expanding_window_splits, rolling_window_splits, train_dev_test_split


PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
MODEL_DIR = "models_saved"

# New: keep robustness runs isolated from your normal saved models.
ROBUSTNESS_DIR = os.path.join(MODEL_DIR, "robustness")
ROBUSTNESS_RESULTS_DIR = os.path.join(RESULTS_DIR, "robustness")


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


@dataclass
class DatasetBundle:
    file_name: str
    task: str
    target_col: str
    df: pd.DataFrame
    dev_df: pd.DataFrame
    test_df: pd.DataFrame
    X_dev: pd.DataFrame
    y_dev: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    folds: List


def ensure_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(ROBUSTNESS_DIR, exist_ok=True)
    os.makedirs(ROBUSTNESS_RESULTS_DIR, exist_ok=True)


def get_expected_model_names(task: str) -> List[str]:
    if task == "regression":
        return list(get_regression_models().keys())
    if task == "classification":
        return list(get_classification_models().keys())
    raise ValueError(f"Unknown task type: {task}")


def get_model_registry(task: str) -> Dict[str, object]:
    if task == "regression":
        return get_regression_models()
    if task == "classification":
        return get_classification_models()
    raise ValueError(f"Unknown task type: {task}")


def get_model_path(
    dataset_name: str,
    model_name: str,
    save_dir: str = MODEL_DIR,
    suffix: str = "",
) -> str:
    safe_dataset = dataset_name.replace(".csv", "")
    safe_suffix = f"__{suffix}" if suffix else ""
    return os.path.join(save_dir, f"{safe_dataset}__{model_name}{safe_suffix}.pkl")


def get_metadata_path(
    dataset_name: str,
    model_name: str,
    save_dir: str = MODEL_DIR,
    suffix: str = "",
) -> str:
    return get_model_path(dataset_name, model_name, save_dir=save_dir, suffix=suffix) + ".meta.json"


def model_exists(
    dataset_name: str,
    model_name: str,
    save_dir: str = MODEL_DIR,
    suffix: str = "",
) -> bool:
    path = get_model_path(dataset_name, model_name, save_dir=save_dir, suffix=suffix)
    return os.path.exists(path) and os.path.getsize(path) > 0


def save_model(
    model,
    dataset_name: str,
    model_name: str,
    save_dir: str = MODEL_DIR,
    suffix: str = "",
    metadata: Optional[Dict] = None,
) -> str:
    os.makedirs(save_dir, exist_ok=True)

    path = get_model_path(dataset_name, model_name, save_dir=save_dir, suffix=suffix)
    meta_path = get_metadata_path(dataset_name, model_name, save_dir=save_dir, suffix=suffix)

    joblib.dump(model, path)

    if metadata is not None:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    return path


def _patch_legacy_model_after_load(model):
    """
    Patch older sklearn model objects loaded from joblib so they remain usable
    in newer sklearn environments.

    This is especially helpful for legacy LogisticRegression objects that may
    be missing newer runtime attributes accessed by predict()/predict_proba().
    """
    try:
        final_est = model.steps[-1][1] if isinstance(model, Pipeline) else model

        if isinstance(final_est, LogisticRegression):
            if not hasattr(final_est, "multi_class"):
                final_est.multi_class = "auto"
            if not hasattr(final_est, "n_jobs"):
                final_est.n_jobs = None
            if not hasattr(final_est, "l1_ratio"):
                final_est.l1_ratio = None
            if not hasattr(final_est, "class_weight"):
                final_est.class_weight = None
            if not hasattr(final_est, "solver"):
                final_est.solver = "lbfgs"
            if not hasattr(final_est, "penalty"):
                final_est.penalty = "l2"
            if not hasattr(final_est, "max_iter"):
                final_est.max_iter = 100
            if not hasattr(final_est, "random_state"):
                final_est.random_state = None

    except Exception as e:
        print(f"[WARN] Legacy model patch failed: {e}")

    return model


def load_model(
    dataset_name: str,
    model_name: str,
    save_dir: str = MODEL_DIR,
    suffix: str = "",
):
    path = get_model_path(dataset_name, model_name, save_dir=save_dir, suffix=suffix)
    model = joblib.load(path)
    model = _patch_legacy_model_after_load(model)
    return model


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
    y_prob: Optional[np.ndarray] = None,
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


def _safe_predict_proba(model, X) -> Optional[np.ndarray]:
    """
    Return positive-class probability if available.
    Falls back to decision_function -> sigmoid-like mapping if needed.
    If neither is available, return None.
    """
    if hasattr(model, "predict_proba"):
        try:
            prob = model.predict_proba(X)
            if prob.ndim == 2 and prob.shape[1] >= 2:
                return prob[:, 1]
            if prob.ndim == 1:
                return prob
        except Exception as e:
            print(f"[WARN] predict_proba failed: {e}")

    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X)
            scores = np.asarray(scores)

            if scores.ndim == 1:
                return 1.0 / (1.0 + np.exp(-scores))

        except Exception as e:
            print(f"[WARN] decision_function fallback failed: {e}")

    return None


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if task == "regression":
        return regression_metrics(y_true, y_pred)
    return classification_metrics(y_true, y_pred, y_prob)


def evaluate_fitted_model(model, X, y, task: str) -> Dict[str, float]:
    y_pred = model.predict(X)

    if task == "regression":
        return regression_metrics(y.values, y_pred)

    y_prob = _safe_predict_proba(model, X)
    return classification_metrics(y.values, y_pred, y_prob)


def build_dataset_bundle(
    file_name: str,
    split_mode: str = "expanding",
    test_size: float = 0.2,
    train_size: int = 504,
    min_train_size: int = 504,
    valid_size: int = 63,
    step_size: int = 21,
) -> DatasetBundle:
    config = DATASET_CONFIG[file_name]
    target_col = config["target"]
    task = config["task"]
    purge = config["purge"]
    embargo = config["embargo"]

    df = load_dataset(file_name)
    dev_df, test_df = train_dev_test_split(df, test_size=test_size)

    X_dev, y_dev = get_feature_target_split(dev_df, target_col)
    X_test, y_test = get_feature_target_split(test_df, target_col)

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
        raise ValueError(f"Unknown split_mode: {split_mode}")

    if len(folds) == 0:
        raise ValueError(
            f"No valid folds for {file_name}. "
            f"Check dataset size and split parameters."
        )

    return DatasetBundle(
        file_name=file_name,
        task=task,
        target_col=target_col,
        df=df,
        dev_df=dev_df,
        test_df=test_df,
        X_dev=X_dev,
        y_dev=y_dev,
        X_test=X_test,
        y_test=y_test,
        folds=folds,
    )


def evaluate_one_fold(
    fresh_model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    task: str,
) -> Dict[str, float]:
    model = clone(fresh_model)
    model.fit(X_train, y_train)
    return evaluate_fitted_model(model, X_valid, y_valid, task)


def summarize_fold_metrics(records: List[Dict[str, float]]) -> Dict[str, float]:
    if len(records) == 0:
        return {}

    df = pd.DataFrame(records)
    summary = {}

    metric_cols = [col for col in df.columns if col not in {"fold", "fold_start", "fold_end"}]
    for col in metric_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            summary[f"{col}_cv_mean"] = float(df[col].mean())
            summary[f"{col}_cv_std"] = float(df[col].std()) if len(df[col]) > 1 else 0.0

    return summary


def _build_model_metadata(
    dataset_name: str,
    model_name: str,
    task: str,
    training_mode: str,
    extra: Optional[Dict] = None,
) -> Dict:
    meta = {
        "dataset": dataset_name,
        "model": model_name,
        "task": task,
        "training_mode": training_mode,
    }
    if extra:
        meta.update(extra)
    return meta


def _standard_train_fit_and_test(
    fresh_model,
    bundle: DatasetBundle,
    model_name: str,
    split_mode: str,
    save_dir: str,
    save_suffix: str = "",
) -> Dict[str, object]:
    fold_records = []

    for i, fold in enumerate(bundle.folds, start=1):
        X_train = bundle.X_dev.iloc[fold.train_idx]
        y_train = bundle.y_dev.iloc[fold.train_idx]
        X_valid = bundle.X_dev.iloc[fold.valid_idx]
        y_valid = bundle.y_dev.iloc[fold.valid_idx]

        metrics = evaluate_one_fold(
            fresh_model=fresh_model,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            task=bundle.task,
        )
        metrics["fold"] = i
        metrics["fold_start"] = int(fold.valid_idx[0])
        metrics["fold_end"] = int(fold.valid_idx[-1])
        fold_records.append(metrics)

    cv_summary = summarize_fold_metrics(fold_records)

    final_model = clone(fresh_model)
    final_model.fit(bundle.X_dev, bundle.y_dev)

    model_path = save_model(
        model=final_model,
        dataset_name=bundle.file_name,
        model_name=model_name,
        save_dir=save_dir,
        suffix=save_suffix,
        metadata=_build_model_metadata(
            dataset_name=bundle.file_name,
            model_name=model_name,
            task=bundle.task,
            training_mode="standard",
            extra={"split_mode": split_mode},
        ),
    )

    test_metrics = evaluate_fitted_model(final_model, bundle.X_test, bundle.y_test, bundle.task)

    row = {
        "dataset": bundle.file_name,
        "task": bundle.task,
        "model": model_name,
        "model_family": "classical_ml",
        "split_mode": split_mode,
        "trained_this_run": True,
        "model_path": model_path,
        "n_dev": len(bundle.dev_df),
        "n_test": len(bundle.test_df),
        "n_folds": len(bundle.folds),
        **cv_summary,
    }
    for k, v in test_metrics.items():
        row[f"test_{k}"] = v

    return row


def run_cv_for_dataset(
    file_name: str,
    split_mode: str = "expanding",
    test_size: float = 0.2,
    train_size: int = 504,
    min_train_size: int = 504,
    valid_size: int = 63,
    step_size: int = 21,
    force_retrain: bool = False,
    save_dir: str = MODEL_DIR,
) -> pd.DataFrame:
    """
    Standard training/evaluation entry point.

    Backward-compatible with your current pipeline, but now allows:
    - force_retrain=True
    - alternate save_dir
    """
    ensure_dirs()

    bundle = build_dataset_bundle(
        file_name=file_name,
        split_mode=split_mode,
        test_size=test_size,
        train_size=train_size,
        min_train_size=min_train_size,
        valid_size=valid_size,
        step_size=step_size,
    )

    model_registry = get_model_registry(bundle.task)
    all_results: List[Dict[str, object]] = []

    for model_name, fresh_model in model_registry.items():
        model_path = get_model_path(bundle.file_name, model_name, save_dir=save_dir)

        if not force_retrain and model_exists(bundle.file_name, model_name, save_dir=save_dir):
            try:
                print(f"Loading existing model: {bundle.file_name} / {model_name}")
                model = load_model(bundle.file_name, model_name, save_dir=save_dir)

                test_metrics = evaluate_fitted_model(
                    model=model,
                    X=bundle.X_test,
                    y=bundle.y_test,
                    task=bundle.task,
                )

                row = {
                    "dataset": bundle.file_name,
                    "task": bundle.task,
                    "model": model_name,
                    "model_family": "classical_ml",
                    "split_mode": split_mode,
                    "trained_this_run": False,
                    "model_path": model_path,
                    "n_dev": len(bundle.dev_df),
                    "n_test": len(bundle.test_df),
                    "n_folds": len(bundle.folds),
                }
                for k, v in test_metrics.items():
                    row[f"test_{k}"] = v

                all_results.append(row)
                continue

            except Exception as e:
                print(f"[WARN] Failed to evaluate loaded model {bundle.file_name} / {model_name}: {e}")
                print("       Treating it as incompatible legacy model. Deleting and retraining.")
                try:
                    os.remove(model_path)
                except OSError as remove_err:
                    print(f"[WARN] Could not remove broken model file: {remove_err}")

        row = _standard_train_fit_and_test(
            fresh_model=fresh_model,
            bundle=bundle,
            model_name=model_name,
            split_mode=split_mode,
            save_dir=save_dir,
            save_suffix="",
        )
        all_results.append(row)

    return pd.DataFrame(all_results)


def run_all_datasets(
    split_mode: str = "expanding",
    force_retrain: bool = False,
    save_dir: str = MODEL_DIR,
    result_filename: Optional[str] = None,
) -> pd.DataFrame:
    ensure_dirs()

    result_frames = []
    for file_name in DATASET_CONFIG:
        print(f"Running dataset: {file_name}")
        result_df = run_cv_for_dataset(
            file_name=file_name,
            split_mode=split_mode,
            force_retrain=force_retrain,
            save_dir=save_dir,
        )
        result_frames.append(result_df)

    final_df = pd.concat(result_frames, axis=0, ignore_index=True)

    if result_filename is None:
        result_filename = f"model_comparison_{split_mode}.csv"

    out_path = os.path.join(RESULTS_DIR, result_filename)
    final_df.to_csv(out_path, index=False)

    print(f"Saved results to {out_path}")
    return final_df


# -------------------------------------------------------------------
# Robustness / Bootstrap Retraining
# -------------------------------------------------------------------

def _fit_bootstrap_model_and_score(
    fresh_model,
    bundle: DatasetBundle,
    model_name: str,
    bootstrap_sample: BootstrapSample,
    bootstrap_id: int,
    split_mode: str,
    save_models: bool,
    save_dir: str,
) -> Dict[str, object]:
    """
    Refit one model on one bootstrap-resampled development set,
    then evaluate on the fixed untouched test set.

    Important:
    - test set remains untouched and NOT bootstrapped
    - dev set is resampled using block / stationary bootstrap
    """
    boot_idx = bootstrap_sample.sample_idx

    X_boot = bundle.X_dev.iloc[boot_idx].reset_index(drop=True)
    y_boot = bundle.y_dev.iloc[boot_idx].reset_index(drop=True)

    final_model = clone(fresh_model)
    final_model.fit(X_boot, y_boot)

    save_suffix = f"bootstrap_{bootstrap_id:04d}" if save_models else ""
    if save_models:
        model_path = save_model(
            model=final_model,
            dataset_name=bundle.file_name,
            model_name=model_name,
            save_dir=save_dir,
            suffix=save_suffix,
            metadata=_build_model_metadata(
                dataset_name=bundle.file_name,
                model_name=model_name,
                task=bundle.task,
                training_mode="bootstrap_retrain",
                extra={
                    "bootstrap_id": bootstrap_id,
                    "bootstrap_method": bootstrap_sample.method,
                    "block_length": bootstrap_sample.block_length,
                    "seed": bootstrap_sample.seed,
                    "split_mode": split_mode,
                },
            ),
        )
    else:
        model_path = ""

    test_metrics = evaluate_fitted_model(final_model, bundle.X_test, bundle.y_test, bundle.task)

    row = {
        "dataset": bundle.file_name,
        "task": bundle.task,
        "model": model_name,
        "model_family": "classical_ml",
        "training_mode": "bootstrap_retrain",
        "split_mode": split_mode,
        "bootstrap_id": int(bootstrap_id),
        "bootstrap_method": bootstrap_sample.method,
        "block_length": int(bootstrap_sample.block_length),
        "bootstrap_seed": int(bootstrap_sample.seed) if bootstrap_sample.seed is not None else np.nan,
        "bootstrap_unique_fraction": float(bootstrap_sample.unique_fraction),
        "trained_this_run": True,
        "model_path": model_path,
        "n_dev_original": len(bundle.dev_df),
        "n_dev_bootstrap": len(X_boot),
        "n_test": len(bundle.test_df),
    }

    for k, v in test_metrics.items():
        row[f"test_{k}"] = v

    return row


def run_bootstrap_for_dataset(
    file_name: str,
    split_mode: str = "expanding",
    test_size: float = 0.2,
    train_size: int = 504,
    min_train_size: int = 504,
    valid_size: int = 63,
    step_size: int = 21,
    n_bootstrap: int = 50,
    bootstrap_method: str = "stationary",
    block_length: Optional[int] = None,
    base_seed: int = 42,
    save_models: bool = False,
    force_retrain: bool = True,
    save_dir: str = ROBUSTNESS_DIR,
) -> pd.DataFrame:
    """
    Bootstrap retraining for one dataset across all model variants.

    This is the main classical-ML robustness retraining entry point.
    """
    ensure_dirs()

    if not force_retrain:
        print("[INFO] force_retrain=False was passed, but bootstrap retraining always refits models.")
        print("       Continuing with refit behavior.")

    bundle = build_dataset_bundle(
        file_name=file_name,
        split_mode=split_mode,
        test_size=test_size,
        train_size=train_size,
        min_train_size=min_train_size,
        valid_size=valid_size,
        step_size=step_size,
    )

    if block_length is None:
        block_length = suggest_block_length(len(bundle.dev_df), rule="sqrt", min_block_length=5)

    model_registry = get_model_registry(bundle.task)
    all_rows: List[Dict[str, object]] = []

    for model_name, fresh_model in model_registry.items():
        print(
            f"[BOOTSTRAP] dataset={bundle.file_name} model={model_name} "
            f"method={bootstrap_method} block_length={block_length} n_bootstrap={n_bootstrap}"
        )

        for b, sample in enumerate(
            iter_bootstrap_samples(
                n_samples=len(bundle.dev_df),
                n_bootstrap=n_bootstrap,
                method=bootstrap_method,
                block_length=block_length,
                base_seed=base_seed,
            ),
            start=1,
        ):
            row = _fit_bootstrap_model_and_score(
                fresh_model=fresh_model,
                bundle=bundle,
                model_name=model_name,
                bootstrap_sample=sample,
                bootstrap_id=b,
                split_mode=split_mode,
                save_models=save_models,
                save_dir=save_dir,
            )
            all_rows.append(row)

    out_df = pd.DataFrame(all_rows)

    out_path = os.path.join(
        ROBUSTNESS_RESULTS_DIR,
        f"classical_bootstrap_{file_name.replace('.csv', '')}_{bootstrap_method}.csv",
    )
    out_df.to_csv(out_path, index=False)
    print(f"Saved bootstrap details to {out_path}")

    summary_df = summarize_bootstrap_results(out_df)
    summary_path = os.path.join(
        ROBUSTNESS_RESULTS_DIR,
        f"classical_bootstrap_summary_{file_name.replace('.csv', '')}_{bootstrap_method}.csv",
    )
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved bootstrap summary to {summary_path}")

    return out_df


def summarize_bootstrap_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate bootstrap retraining results by dataset/model.

    Produces mean/std and percentile CI for each numeric test metric.
    """
    if df.empty:
        return pd.DataFrame()

    group_cols = ["dataset", "task", "model", "model_family", "training_mode", "bootstrap_method", "block_length"]

    numeric_cols = [
        col for col in df.columns
        if col.startswith("test_") and pd.api.types.is_numeric_dtype(df[col])
    ]

    rows: List[Dict[str, object]] = []

    for keys, grp in df.groupby(group_cols, dropna=False):
        row = {col: val for col, val in zip(group_cols, keys)}
        row["n_bootstrap"] = int(len(grp))

        for col in numeric_cols:
            vals = grp[col].dropna().astype(float).values
            if len(vals) == 0:
                continue

            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{col}_median"] = float(np.median(vals))
            row[f"{col}_p05"] = float(np.quantile(vals, 0.05))
            row[f"{col}_p25"] = float(np.quantile(vals, 0.25))
            row[f"{col}_p75"] = float(np.quantile(vals, 0.75))
            row[f"{col}_p95"] = float(np.quantile(vals, 0.95))
            row[f"{col}_ci_low_95"] = float(np.quantile(vals, 0.025))
            row[f"{col}_ci_high_95"] = float(np.quantile(vals, 0.975))

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["dataset", "model"]).reset_index(drop=True)


def run_bootstrap_all_datasets(
    split_mode: str = "expanding",
    n_bootstrap: int = 50,
    bootstrap_method: str = "stationary",
    block_length: Optional[int] = None,
    base_seed: int = 42,
    save_models: bool = False,
    save_dir: str = ROBUSTNESS_DIR,
) -> pd.DataFrame:
    ensure_dirs()

    frames = []
    for file_name in DATASET_CONFIG:
        df = run_bootstrap_for_dataset(
            file_name=file_name,
            split_mode=split_mode,
            n_bootstrap=n_bootstrap,
            bootstrap_method=bootstrap_method,
            block_length=block_length,
            base_seed=base_seed,
            save_models=save_models,
            save_dir=save_dir,
        )
        frames.append(df)

    final_df = pd.concat(frames, axis=0, ignore_index=True)

    out_path = os.path.join(
        ROBUSTNESS_RESULTS_DIR,
        f"classical_bootstrap_all_{bootstrap_method}.csv",
    )
    final_df.to_csv(out_path, index=False)
    print(f"Saved combined bootstrap details to {out_path}")

    summary_df = summarize_bootstrap_results(final_df)
    summary_path = os.path.join(
        ROBUSTNESS_RESULTS_DIR,
        f"classical_bootstrap_all_summary_{bootstrap_method}.csv",
    )
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved combined bootstrap summary to {summary_path}")

    return final_df


# -------------------------------------------------------------------
# Optional walk-forward detail export
# -------------------------------------------------------------------

def run_walk_forward_detail_for_dataset(
    file_name: str,
    split_mode: str = "expanding",
    test_size: float = 0.2,
    train_size: int = 504,
    min_train_size: int = 504,
    valid_size: int = 63,
    step_size: int = 21,
) -> pd.DataFrame:
    """
    Export fold-level CV metrics for every model.
    Useful for walk-forward stability plots/tables.
    """
    ensure_dirs()

    bundle = build_dataset_bundle(
        file_name=file_name,
        split_mode=split_mode,
        test_size=test_size,
        train_size=train_size,
        min_train_size=min_train_size,
        valid_size=valid_size,
        step_size=step_size,
    )

    model_registry = get_model_registry(bundle.task)
    rows: List[Dict[str, object]] = []

    for model_name, fresh_model in model_registry.items():
        for i, fold in enumerate(bundle.folds, start=1):
            X_train = bundle.X_dev.iloc[fold.train_idx]
            y_train = bundle.y_dev.iloc[fold.train_idx]
            X_valid = bundle.X_dev.iloc[fold.valid_idx]
            y_valid = bundle.y_dev.iloc[fold.valid_idx]

            model = clone(fresh_model)
            model.fit(X_train, y_train)

            fold_metrics = evaluate_fitted_model(model, X_valid, y_valid, bundle.task)

            row = {
                "dataset": bundle.file_name,
                "task": bundle.task,
                "model": model_name,
                "model_family": "classical_ml",
                "split_mode": split_mode,
                "fold": i,
                "train_start_idx": int(fold.train_idx[0]),
                "train_end_idx": int(fold.train_idx[-1]),
                "valid_start_idx": int(fold.valid_idx[0]),
                "valid_end_idx": int(fold.valid_idx[-1]),
                "train_size_effective": int(len(fold.train_idx)),
                "valid_size": int(len(fold.valid_idx)),
                "valid_start_date": str(bundle.dev_df.iloc[fold.valid_idx[0]]["Date"].date()),
                "valid_end_date": str(bundle.dev_df.iloc[fold.valid_idx[-1]]["Date"].date()),
            }
            row.update(fold_metrics)
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(
        ROBUSTNESS_RESULTS_DIR,
        f"classical_walk_forward_detail_{file_name.replace('.csv', '')}_{split_mode}.csv",
    )
    out_df.to_csv(out_path, index=False)
    print(f"Saved walk-forward detail to {out_path}")
    return out_df


if __name__ == "__main__":
    # Standard run:
    # df = run_all_datasets(split_mode="expanding", force_retrain=False)

    # Bootstrap retraining run:
    # df = run_bootstrap_all_datasets(
    #     split_mode="expanding",
    #     n_bootstrap=30,
    #     bootstrap_method="stationary",
    #     block_length=None,
    #     base_seed=42,
    #     save_models=False,
    # )

    df = run_all_datasets(split_mode="expanding", force_retrain=False)
    print(df)