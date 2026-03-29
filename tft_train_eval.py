# tft_train_eval.py
from __future__ import annotations

import json
import os
import pickle
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from bootstrap_utils import BootstrapSample, iter_bootstrap_samples, suggest_block_length
from splitter import expanding_window_splits, train_dev_test_split


warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
MODEL_DIR = "models_saved_tft"
WORK_DIR = os.path.join(MODEL_DIR, "tft_workdir")

ROBUSTNESS_DIR = os.path.join(MODEL_DIR, "robustness")
ROBUSTNESS_RESULTS_DIR = os.path.join(RESULTS_DIR, "robustness")
ROBUSTNESS_WORK_DIR = os.path.join(ROBUSTNESS_DIR, "tft_workdir")

RUN_TFT_CLASSIFICATION = True
RUN_TFT_LIGHT_CV = True
SKIP_FINAL_TRAIN_IF_ARTIFACTS_EXIST = True

DEFAULT_TEST_SIZE = 0.2
DEFAULT_MIN_TRAIN_SIZE = 504
DEFAULT_VALID_SIZE = 63
DEFAULT_STEP_SIZE = 21
FINAL_VALID_SIZE = 63
CLASSIFICATION_THRESHOLD = 0.5
CV_HISTORICAL_STRIDE = 5
TEST_HISTORICAL_STRIDE = 1

MAIN_RESULTS_FILENAME = "tft_model_comparison_expanding.csv"
EXPLORATORY_RESULTS_FILENAME = "tft_classification_exploratory_expanding.csv"

TFT_DATASET_CONFIG = {
    "dataset_return.csv": {
        "target": "y_return",
        "task": "regression",
        "purge": 1,
        "embargo": 1,
    },
    "dataset_volatility.csv": {
        "target": "y_vol",
        "task": "regression",
        "purge": 5,
        "embargo": 5,
    },
    "dataset_direction.csv": {
        "target": "y_direction",
        "task": "classification",
        "purge": 1,
        "embargo": 1,
    },
    "dataset_regime.csv": {
        "target": "y_regime",
        "task": "classification",
        "purge": 5,
        "embargo": 5,
    },
}

TFT_MODEL_CONFIGS = {
    "tft_small": {
        "input_chunk_length": 20,
        "output_chunk_length": 1,
        "hidden_size": 8,
        "lstm_layers": 1,
        "num_attention_heads": 2,
        "dropout": 0.1,
        "hidden_continuous_size": 4,
        "batch_size": 512,
        "n_epochs": 5,
        "learning_rate": 1e-3,
    },
    "tft_medium": {
        "input_chunk_length": 30,
        "output_chunk_length": 1,
        "hidden_size": 16,
        "lstm_layers": 1,
        "num_attention_heads": 2,
        "dropout": 0.1,
        "hidden_continuous_size": 8,
        "batch_size": 512,
        "n_epochs": 10,
        "learning_rate": 1e-3,
    },
    "tft_large": {
        "input_chunk_length": 40,
        "output_chunk_length": 1,
        "hidden_size": 24,
        "lstm_layers": 2,
        "num_attention_heads": 4,
        "dropout": 0.2,
        "hidden_continuous_size": 12,
        "batch_size": 512,
        "n_epochs": 15,
        "learning_rate": 5e-4,
    },
}


@dataclass
class PreparedSeries:
    train_target_scaled: TimeSeries
    train_cov_scaled: TimeSeries
    valid_target_scaled: TimeSeries
    valid_cov_scaled: TimeSeries
    target_scaler: Optional[Scaler]
    cov_scaler: Scaler


@dataclass
class LoadedArtifacts:
    model: TFTModel
    task: str
    target_col: str
    feature_cols: List[str]
    cfg: Dict
    cov_scaler: Scaler
    target_scaler: Optional[Scaler]
    metadata: Dict


@dataclass
class TFTBundle:
    file_name: str
    df: pd.DataFrame
    dev_df: pd.DataFrame
    test_df: pd.DataFrame
    target_col: str
    task: str
    feature_cols: List[str]
    folds: List


def ensure_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(ROBUSTNESS_DIR, exist_ok=True)
    os.makedirs(ROBUSTNESS_RESULTS_DIR, exist_ok=True)
    os.makedirs(ROBUSTNESS_WORK_DIR, exist_ok=True)


def load_dataset(file_name: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing processed dataset: {path}")

    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"Dataset must contain a 'Date' column: {file_name}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    numeric_cols = [col for col in df.columns if col != "Date"]
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].astype(np.float32)

    return df


def with_local_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["time_idx"] = np.arange(len(out), dtype=int)
    return out


def get_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    exclude = {"Date", "time_idx", target_col}
    feature_cols = [col for col in df.columns if col not in exclude]
    if not feature_cols:
        raise ValueError(f"No feature columns found for target={target_col}")
    return feature_cols


def build_series_from_df(df: pd.DataFrame, value_cols: List[str]) -> TimeSeries:
    local_df = with_local_time_idx(df)
    local_df[value_cols] = local_df[value_cols].astype(np.float32)
    return TimeSeries.from_dataframe(
        local_df,
        time_col="time_idx",
        value_cols=value_cols,
        fill_missing_dates=False,
    ).astype(np.float32)


def ensure_series_float32(series: TimeSeries) -> TimeSeries:
    return series.astype(np.float32)


def build_target_series(df: pd.DataFrame, target_col: str) -> TimeSeries:
    return build_series_from_df(df, [target_col])


def build_covariate_series(df: pd.DataFrame, feature_cols: List[str]) -> TimeSeries:
    return build_series_from_df(df, feature_cols)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "oos_r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= threshold).astype(int)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "positive_rate_pred": float(np.mean(y_pred)),
        "positive_rate_true": float(np.mean(y_true)),
        "n_positive_true": int(np.sum(y_true == 1)),
        "n_negative_true": int(np.sum(y_true == 0)),
    }

    unique_classes = np.unique(y_true)
    out["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if len(unique_classes) == 2 else np.nan
    return out


def summarize_fold_metrics(records: List[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        return {}

    df = pd.DataFrame(records)
    summary: Dict[str, float] = {}

    metric_cols = [
        col for col in df.columns
        if col not in {"fold", "fold_start", "fold_end"} and pd.api.types.is_numeric_dtype(df[col])
    ]

    for col in metric_cols:
        summary[f"{col}_cv_mean"] = float(df[col].mean())
        summary[f"{col}_cv_std"] = float(df[col].std()) if len(df[col]) > 1 else 0.0

    return summary


def model_path(dataset_name: str, model_name: str, save_dir: str = MODEL_DIR, suffix: str = "") -> str:
    safe_dataset = dataset_name.replace(".csv", "")
    safe_suffix = f"__{suffix}" if suffix else ""
    return os.path.join(save_dir, f"{safe_dataset}__{model_name}{safe_suffix}.pt")


def meta_path(dataset_name: str, model_name: str, save_dir: str = MODEL_DIR, suffix: str = "") -> str:
    return model_path(dataset_name, model_name, save_dir=save_dir, suffix=suffix) + ".meta.json"


def cov_scaler_path(dataset_name: str, model_name: str, save_dir: str = MODEL_DIR, suffix: str = "") -> str:
    return model_path(dataset_name, model_name, save_dir=save_dir, suffix=suffix) + ".cov_scaler.pkl"


def target_scaler_path(dataset_name: str, model_name: str, save_dir: str = MODEL_DIR, suffix: str = "") -> str:
    return model_path(dataset_name, model_name, save_dir=save_dir, suffix=suffix) + ".target_scaler.pkl"


def save_pickle(obj, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_early_stopping(cfg: Dict) -> EarlyStopping:
    patience = 2 if cfg["n_epochs"] <= 8 else 3
    return EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
    )


def build_tft_model(
    model_name: str,
    cfg: Dict,
    task: str,
    work_subdir: str,
) -> TFTModel:
    if task == "regression":
        loss_fn = torch.nn.MSELoss()
    elif task == "classification":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported task: {task}")

    early_stop = build_early_stopping(cfg)

    return TFTModel(
        input_chunk_length=cfg["input_chunk_length"],
        output_chunk_length=cfg["output_chunk_length"],
        hidden_size=cfg["hidden_size"],
        lstm_layers=cfg["lstm_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        dropout=cfg["dropout"],
        hidden_continuous_size=cfg["hidden_continuous_size"],
        add_relative_index=True,
        likelihood=None,
        loss_fn=loss_fn,
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        optimizer_kwargs={"lr": cfg["learning_rate"]},
        random_state=42,
        model_name=model_name,
        work_dir=work_subdir,
        save_checkpoints=True,
        force_reset=True,
        log_tensorboard=False,
        pl_trainer_kwargs={
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
            "precision": "bf16-mixed" if torch.cuda.is_available() else "32-true",
            "gradient_clip_val": 0.5,
            "enable_progress_bar": True,
            "logger": False,
            "callbacks": [early_stop],
        },
    )


def prepare_scaled_series(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    task: str,
) -> PreparedSeries:
    train_target = build_target_series(train_df, target_col)
    valid_target = build_target_series(valid_df, target_col)
    train_cov = build_covariate_series(train_df, feature_cols)
    valid_cov = build_covariate_series(valid_df, feature_cols)

    cov_scaler = Scaler()
    train_cov_scaled = ensure_series_float32(cov_scaler.fit_transform(train_cov))
    valid_cov_scaled = ensure_series_float32(cov_scaler.transform(valid_cov))

    if task == "regression":
        target_scaler: Optional[Scaler] = Scaler()
        train_target_scaled = ensure_series_float32(target_scaler.fit_transform(train_target))
        valid_target_scaled = ensure_series_float32(target_scaler.transform(valid_target))
    elif task == "classification":
        target_scaler = None
        train_target_scaled = ensure_series_float32(train_target)
        valid_target_scaled = ensure_series_float32(valid_target)
    else:
        raise ValueError(f"Unsupported task: {task}")

    return PreparedSeries(
        train_target_scaled=train_target_scaled,
        train_cov_scaled=train_cov_scaled,
        valid_target_scaled=valid_target_scaled,
        valid_cov_scaled=valid_cov_scaled,
        target_scaler=target_scaler,
        cov_scaler=cov_scaler,
    )


def get_historical_regression_predictions(
    model: TFTModel,
    target_scaler: Scaler,
    combined_target_scaled: TimeSeries,
    combined_cov_scaled: TimeSeries,
    combined_target_unscaled: TimeSeries,
    start_time: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    preds_scaled = model.historical_forecasts(
        series=combined_target_scaled,
        future_covariates=combined_cov_scaled,
        start=start_time,
        forecast_horizon=1,
        stride=stride,
        retrain=False,
        last_points_only=True,
        verbose=False,
    )

    preds = target_scaler.inverse_transform(preds_scaled)
    pred_values = preds.values(copy=False).flatten()
    actual_values = combined_target_unscaled.slice_intersect(preds).values(copy=False).flatten()
    return actual_values, pred_values


def get_historical_classification_predictions(
    model: TFTModel,
    combined_target: TimeSeries,
    combined_cov_scaled: TimeSeries,
    start_time: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds_logits = model.historical_forecasts(
        series=combined_target,
        future_covariates=combined_cov_scaled,
        start=start_time,
        forecast_horizon=1,
        stride=stride,
        retrain=False,
        last_points_only=True,
        verbose=False,
    )

    logits = preds_logits.values(copy=False).flatten()
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_true = combined_target.slice_intersect(preds_logits).values(copy=False).flatten().astype(int)
    return y_true, logits, probs


def split_dev_for_final_training(dev_df: pd.DataFrame, final_valid_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(dev_df) <= final_valid_size:
        raise ValueError(
            f"Development set too small for final validation split: len(dev_df)={len(dev_df)}, "
            f"final_valid_size={final_valid_size}"
        )

    split_idx = len(dev_df) - final_valid_size
    final_train_df = dev_df.iloc[:split_idx].reset_index(drop=True)
    final_valid_df = dev_df.iloc[split_idx:].reset_index(drop=True)
    return final_train_df, final_valid_df


def save_tft_artifacts(
    model: TFTModel,
    dataset_name: str,
    model_name: str,
    task: str,
    target_col: str,
    feature_cols: List[str],
    cfg: Dict,
    cov_scaler: Scaler,
    target_scaler: Optional[Scaler],
    save_dir: str = MODEL_DIR,
    suffix: str = "",
    extra_meta: Optional[Dict] = None,
) -> Dict[str, str]:
    os.makedirs(save_dir, exist_ok=True)

    saved_model_path = model_path(dataset_name, model_name, save_dir=save_dir, suffix=suffix)
    saved_cov_scaler_path = cov_scaler_path(dataset_name, model_name, save_dir=save_dir, suffix=suffix)
    saved_target_scaler_path = target_scaler_path(dataset_name, model_name, save_dir=save_dir, suffix=suffix)

    model.save(saved_model_path)
    save_pickle(cov_scaler, saved_cov_scaler_path)

    if task == "regression":
        if target_scaler is None:
            raise ValueError("Regression task must save a target scaler.")
        save_pickle(target_scaler, saved_target_scaler_path)
    else:
        if os.path.exists(saved_target_scaler_path):
            os.remove(saved_target_scaler_path)

    metadata = {
        "dataset": dataset_name,
        "model": model_name,
        "task": task,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "config": cfg,
        "run_tft_classification": RUN_TFT_CLASSIFICATION,
        "run_tft_light_cv": RUN_TFT_LIGHT_CV,
        "classification_threshold": CLASSIFICATION_THRESHOLD if task == "classification" else None,
        "has_target_scaler": bool(task == "regression"),
    }
    if extra_meta:
        metadata.update(extra_meta)

    with open(meta_path(dataset_name, model_name, save_dir=save_dir, suffix=suffix), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return {
        "model_path": saved_model_path,
        "cov_scaler_path": saved_cov_scaler_path,
        "target_scaler_path": saved_target_scaler_path if task == "regression" else "",
    }


def final_artifacts_exist(
    dataset_name: str,
    model_name: str,
    task: str,
    save_dir: str = MODEL_DIR,
    suffix: str = "",
) -> bool:
    model_exists = os.path.exists(model_path(dataset_name, model_name, save_dir=save_dir, suffix=suffix))
    meta_exists = os.path.exists(meta_path(dataset_name, model_name, save_dir=save_dir, suffix=suffix))
    cov_exists = os.path.exists(cov_scaler_path(dataset_name, model_name, save_dir=save_dir, suffix=suffix))

    if task == "regression":
        target_exists = os.path.exists(target_scaler_path(dataset_name, model_name, save_dir=save_dir, suffix=suffix))
        return model_exists and meta_exists and cov_exists and target_exists

    return model_exists and meta_exists and cov_exists


def load_tft_artifacts(
    dataset_name: str,
    model_name: str,
    expected_task: str,
    expected_target_col: str,
    expected_feature_cols: List[str],
    expected_cfg: Dict,
    save_dir: str = MODEL_DIR,
    suffix: str = "",
) -> LoadedArtifacts:
    mp = model_path(dataset_name, model_name, save_dir=save_dir, suffix=suffix)
    metap = meta_path(dataset_name, model_name, save_dir=save_dir, suffix=suffix)
    covp = cov_scaler_path(dataset_name, model_name, save_dir=save_dir, suffix=suffix)
    targp = target_scaler_path(dataset_name, model_name, save_dir=save_dir, suffix=suffix)

    if not final_artifacts_exist(dataset_name, model_name, expected_task, save_dir=save_dir, suffix=suffix):
        raise FileNotFoundError(f"Incomplete artifacts for {dataset_name} / {model_name}")

    with open(metap, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    task = metadata.get("task")
    target_col = metadata.get("target_col")
    feature_cols = metadata.get("feature_cols")
    cfg = metadata.get("config")

    if task != expected_task:
        raise ValueError(
            f"Saved task mismatch for {dataset_name} / {model_name}: "
            f"saved={task}, expected={expected_task}"
        )
    if target_col != expected_target_col:
        raise ValueError(
            f"Saved target mismatch for {dataset_name} / {model_name}: "
            f"saved={target_col}, expected={expected_target_col}"
        )
    if feature_cols != expected_feature_cols:
        raise ValueError(
            f"Saved feature columns mismatch for {dataset_name} / {model_name}.\n"
            f"saved={feature_cols}\nexpected={expected_feature_cols}"
        )
    if cfg != expected_cfg:
        raise ValueError(
            f"Saved config mismatch for {dataset_name} / {model_name}.\n"
            f"saved={cfg}\nexpected={expected_cfg}"
        )

    model = TFTModel.load(mp)
    cov_scaler = load_pickle(covp)
    target_scaler = load_pickle(targp) if expected_task == "regression" else None

    return LoadedArtifacts(
        model=model,
        task=task,
        target_col=target_col,
        feature_cols=feature_cols,
        cfg=cfg,
        cov_scaler=cov_scaler,
        target_scaler=target_scaler,
        metadata=metadata,
    )


def fit_one_model(
    dataset_name: str,
    model_name: str,
    cfg: Dict,
    task: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    work_dir: str,
) -> Tuple[TFTModel, PreparedSeries]:
    prepared = prepare_scaled_series(
        train_df=train_df,
        valid_df=valid_df,
        target_col=target_col,
        feature_cols=feature_cols,
        task=task,
    )

    model = build_tft_model(
        model_name=model_name,
        cfg=cfg,
        task=task,
        work_subdir=work_dir,
    )

    fit_kwargs = {
        "series": prepared.train_target_scaled,
        "future_covariates": prepared.train_cov_scaled,
        "val_series": prepared.valid_target_scaled,
        "val_future_covariates": prepared.valid_cov_scaled,
        "verbose": False,
    }

    dataloader_kwargs = {
        "num_workers": 8 if torch.cuda.is_available() else 0,
        "pin_memory": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available() and dataloader_kwargs["num_workers"] > 0:
        dataloader_kwargs["persistent_workers"] = True

    try:
        model.fit(**fit_kwargs, dataloader_kwargs=dataloader_kwargs)
    except TypeError:
        model.fit(**fit_kwargs)

    return model, prepared


def build_tft_bundle(
    file_name: str,
    test_size: float = DEFAULT_TEST_SIZE,
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
    valid_size: int = DEFAULT_VALID_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
) -> TFTBundle:
    config = TFT_DATASET_CONFIG[file_name]
    target_col = config["target"]
    task = config["task"]
    purge = config["purge"]
    embargo = config["embargo"]

    df = load_dataset(file_name)
    feature_cols = get_feature_columns(df, target_col)

    dev_df, test_df = train_dev_test_split(df, test_size=test_size)

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

    if len(folds) == 0:
        raise ValueError(
            f"No valid folds for {file_name}. "
            f"Check dataset size and split parameters."
        )

    return TFTBundle(
        file_name=file_name,
        df=df,
        dev_df=dev_df,
        test_df=test_df,
        target_col=target_col,
        task=task,
        feature_cols=feature_cols,
        folds=folds,
    )


def fit_fold_and_score(
    dataset_name: str,
    model_name: str,
    cfg: Dict,
    task: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    work_dir: str,
) -> Dict[str, float]:
    model, prepared = fit_one_model(
        dataset_name=dataset_name,
        model_name=model_name,
        cfg=cfg,
        task=task,
        train_df=train_df,
        valid_df=valid_df,
        target_col=target_col,
        feature_cols=feature_cols,
        work_dir=work_dir,
    )

    combined_df = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)

    if task == "regression":
        combined_target_scaled = ensure_series_float32(
            prepared.target_scaler.transform(build_target_series(combined_df, target_col))
        )
        combined_cov_scaled = ensure_series_float32(
            prepared.cov_scaler.transform(build_covariate_series(combined_df, feature_cols))
        )
        combined_target_unscaled = build_target_series(combined_df, target_col)

        start_time = len(train_df)
        y_true, y_pred = get_historical_regression_predictions(
            model=model,
            target_scaler=prepared.target_scaler,
            combined_target_scaled=combined_target_scaled,
            combined_cov_scaled=combined_cov_scaled,
            combined_target_unscaled=combined_target_unscaled,
            start_time=start_time,
            stride=CV_HISTORICAL_STRIDE,
        )
        return regression_metrics(y_true, y_pred)

    combined_target = build_target_series(combined_df, target_col)
    combined_cov_scaled = ensure_series_float32(
        prepared.cov_scaler.transform(build_covariate_series(combined_df, feature_cols))
    )
    start_time = len(train_df)
    y_true, _, y_prob = get_historical_classification_predictions(
        model=model,
        combined_target=combined_target,
        combined_cov_scaled=combined_cov_scaled,
        start_time=start_time,
        stride=CV_HISTORICAL_STRIDE,
    )
    return classification_metrics(y_true, y_prob)


def final_fit_and_test(
    dataset_name: str,
    model_name: str,
    cfg: Dict,
    task: str,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    save_dir: str = MODEL_DIR,
    suffix: str = "",
    work_dir: str = WORK_DIR,
    extra_meta: Optional[Dict] = None,
) -> Dict[str, object]:
    final_train_df, final_valid_df = split_dev_for_final_training(dev_df, FINAL_VALID_SIZE)

    model, prepared = fit_one_model(
        dataset_name=dataset_name,
        model_name=model_name,
        cfg=cfg,
        task=task,
        train_df=final_train_df,
        valid_df=final_valid_df,
        target_col=target_col,
        feature_cols=feature_cols,
        work_dir=work_dir,
    )

    saved_paths = save_tft_artifacts(
        model=model,
        dataset_name=dataset_name,
        model_name=model_name,
        task=task,
        target_col=target_col,
        feature_cols=feature_cols,
        cfg=cfg,
        cov_scaler=prepared.cov_scaler,
        target_scaler=prepared.target_scaler,
        save_dir=save_dir,
        suffix=suffix,
        extra_meta=extra_meta,
    )

    combined_df = pd.concat([dev_df, test_df], axis=0).reset_index(drop=True)

    if task == "regression":
        combined_target_scaled = ensure_series_float32(
            prepared.target_scaler.transform(build_target_series(combined_df, target_col))
        )
        combined_cov_scaled = ensure_series_float32(
            prepared.cov_scaler.transform(build_covariate_series(combined_df, feature_cols))
        )
        combined_target_unscaled = build_target_series(combined_df, target_col)

        start_time = len(dev_df)
        y_true, y_pred = get_historical_regression_predictions(
            model=model,
            target_scaler=prepared.target_scaler,
            combined_target_scaled=combined_target_scaled,
            combined_cov_scaled=combined_cov_scaled,
            combined_target_unscaled=combined_target_unscaled,
            start_time=start_time,
            stride=TEST_HISTORICAL_STRIDE,
        )
        test_metrics = regression_metrics(y_true, y_pred)
    else:
        combined_target = build_target_series(combined_df, target_col)
        combined_cov_scaled = ensure_series_float32(
            prepared.cov_scaler.transform(build_covariate_series(combined_df, feature_cols))
        )
        start_time = len(dev_df)
        y_true, _, y_prob = get_historical_classification_predictions(
            model=model,
            combined_target=combined_target,
            combined_cov_scaled=combined_cov_scaled,
            start_time=start_time,
            stride=TEST_HISTORICAL_STRIDE,
        )
        test_metrics = classification_metrics(y_true, y_prob)

    return {
        "model_path": saved_paths["model_path"],
        "cov_scaler_path": saved_paths["cov_scaler_path"],
        "target_scaler_path": saved_paths["target_scaler_path"],
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }


def evaluate_loaded_artifacts_on_test(
    artifacts: LoadedArtifacts,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, float]:
    combined_df = pd.concat([dev_df, test_df], axis=0).reset_index(drop=True)

    if artifacts.task == "regression":
        combined_target_scaled = ensure_series_float32(
            artifacts.target_scaler.transform(build_target_series(combined_df, artifacts.target_col))
        )
        combined_cov_scaled = ensure_series_float32(
            artifacts.cov_scaler.transform(build_covariate_series(combined_df, artifacts.feature_cols))
        )
        combined_target_unscaled = build_target_series(combined_df, artifacts.target_col)

        start_time = len(dev_df)
        y_true, y_pred = get_historical_regression_predictions(
            model=artifacts.model,
            target_scaler=artifacts.target_scaler,
            combined_target_scaled=combined_target_scaled,
            combined_cov_scaled=combined_cov_scaled,
            combined_target_unscaled=combined_target_unscaled,
            start_time=start_time,
            stride=TEST_HISTORICAL_STRIDE,
        )
        return regression_metrics(y_true, y_pred)

    combined_target = build_target_series(combined_df, artifacts.target_col)
    combined_cov_scaled = ensure_series_float32(
        artifacts.cov_scaler.transform(build_covariate_series(combined_df, artifacts.feature_cols))
    )
    start_time = len(dev_df)
    y_true, _, y_prob = get_historical_classification_predictions(
        model=artifacts.model,
        combined_target=combined_target,
        combined_cov_scaled=combined_cov_scaled,
        start_time=start_time,
        stride=TEST_HISTORICAL_STRIDE,
    )
    return classification_metrics(y_true, y_prob)


def run_tft_cv_for_dataset(
    file_name: str,
    test_size: float = DEFAULT_TEST_SIZE,
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
    valid_size: int = DEFAULT_VALID_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
    force_retrain: bool = False,
    save_dir: str = MODEL_DIR,
    result_dir: str = RESULTS_DIR,
    work_dir: str = WORK_DIR,
) -> pd.DataFrame:
    ensure_dirs()

    bundle = build_tft_bundle(
        file_name=file_name,
        test_size=test_size,
        min_train_size=min_train_size,
        valid_size=valid_size,
        step_size=step_size,
    )

    rows = []

    for model_name, cfg in TFT_MODEL_CONFIGS.items():
        if bundle.task == "classification" and not RUN_TFT_CLASSIFICATION:
            continue

        artifacts_exist = final_artifacts_exist(
            dataset_name=bundle.file_name,
            model_name=model_name,
            task=bundle.task,
            save_dir=save_dir,
        )

        if artifacts_exist and SKIP_FINAL_TRAIN_IF_ARTIFACTS_EXIST and not force_retrain:
            print(f"[SKIP TRAIN] {bundle.file_name} / {model_name} using saved final artifacts")
            artifacts = load_tft_artifacts(
                dataset_name=bundle.file_name,
                model_name=model_name,
                expected_task=bundle.task,
                expected_target_col=bundle.target_col,
                expected_feature_cols=bundle.feature_cols,
                expected_cfg=cfg,
                save_dir=save_dir,
            )

            test_metrics = evaluate_loaded_artifacts_on_test(
                artifacts=artifacts,
                dev_df=bundle.dev_df,
                test_df=bundle.test_df,
            )

            row = {
                "dataset": bundle.file_name,
                "task": bundle.task,
                "model": model_name,
                "model_family": "tft",
                "trained_this_run": False,
                "model_path": model_path(bundle.file_name, model_name, save_dir=save_dir),
                "n_dev": len(bundle.dev_df),
                "n_test": len(bundle.test_df),
                "n_folds": len(bundle.folds),
                **test_metrics,
            }
            row = {k if k.startswith("test_") or k in {"dataset","task","model","model_family","trained_this_run","model_path","n_dev","n_test","n_folds"} else f"test_{k}": v for k, v in row.items()}
            rows.append(row)
            continue

        fold_records = []
        if RUN_TFT_LIGHT_CV:
            usable_folds = bundle.folds[-min(5, len(bundle.folds)):]
            for i, fold in enumerate(usable_folds, start=1):
                train_df = bundle.dev_df.iloc[fold.train_idx].reset_index(drop=True)
                valid_df = bundle.dev_df.iloc[fold.valid_idx].reset_index(drop=True)

                metrics = fit_fold_and_score(
                    dataset_name=bundle.file_name,
                    model_name=model_name,
                    cfg=cfg,
                    task=bundle.task,
                    train_df=train_df,
                    valid_df=valid_df,
                    target_col=bundle.target_col,
                    feature_cols=bundle.feature_cols,
                    work_dir=work_dir,
                )
                metrics["fold"] = i
                metrics["fold_start"] = int(fold.valid_idx[0])
                metrics["fold_end"] = int(fold.valid_idx[-1])
                fold_records.append(metrics)

        cv_summary = summarize_fold_metrics(fold_records)

        final_results = final_fit_and_test(
            dataset_name=bundle.file_name,
            model_name=model_name,
            cfg=cfg,
            task=bundle.task,
            dev_df=bundle.dev_df,
            test_df=bundle.test_df,
            target_col=bundle.target_col,
            feature_cols=bundle.feature_cols,
            save_dir=save_dir,
            work_dir=work_dir,
            extra_meta={
                "training_mode": "standard",
                "test_size": test_size,
                "min_train_size": min_train_size,
                "valid_size": valid_size,
                "step_size": step_size,
            },
        )

        row = {
            "dataset": bundle.file_name,
            "task": bundle.task,
            "model": model_name,
            "model_family": "tft",
            "trained_this_run": True,
            "model_path": final_results["model_path"],
            "n_dev": len(bundle.dev_df),
            "n_test": len(bundle.test_df),
            "n_folds": len(bundle.folds),
            **cv_summary,
        }
        for k, v in final_results.items():
            if k.startswith("test_"):
                row[k] = v

        rows.append(row)

    out_df = pd.DataFrame(rows)
    os.makedirs(result_dir, exist_ok=True)

    out_path = os.path.join(
        result_dir,
        f"tft_model_comparison_{file_name.replace('.csv', '')}.csv",
    )
    out_df.to_csv(out_path, index=False)
    print(f"Saved TFT results to {out_path}")
    return out_df


def run_all_tft_datasets(
    force_retrain: bool = False,
    save_dir: str = MODEL_DIR,
    result_filename: str = MAIN_RESULTS_FILENAME,
    result_dir: str = RESULTS_DIR,
    work_dir: str = WORK_DIR,
) -> pd.DataFrame:
    ensure_dirs()

    frames = []
    for file_name in TFT_DATASET_CONFIG:
        print(f"[TFT] Running dataset: {file_name}")
        df = run_tft_cv_for_dataset(
            file_name=file_name,
            force_retrain=force_retrain,
            save_dir=save_dir,
            result_dir=result_dir,
            work_dir=work_dir,
        )
        frames.append(df)

    final_df = pd.concat(frames, axis=0, ignore_index=True)
    out_path = os.path.join(result_dir, result_filename)
    final_df.to_csv(out_path, index=False)
    print(f"Saved main TFT results to {out_path}")
    return final_df


# -------------------------------------------------------------------
# Bootstrap retraining
# -------------------------------------------------------------------

def fit_bootstrap_tft_and_score(
    bundle: TFTBundle,
    model_name: str,
    cfg: Dict,
    bootstrap_sample: BootstrapSample,
    bootstrap_id: int,
    save_models: bool,
    save_dir: str,
    work_dir: str,
) -> Dict[str, object]:
    boot_idx = bootstrap_sample.sample_idx
    boot_dev_df = bundle.dev_df.iloc[boot_idx].reset_index(drop=True)

    final_results = final_fit_and_test(
        dataset_name=bundle.file_name,
        model_name=model_name,
        cfg=cfg,
        task=bundle.task,
        dev_df=boot_dev_df,
        test_df=bundle.test_df,
        target_col=bundle.target_col,
        feature_cols=bundle.feature_cols,
        save_dir=save_dir if save_models else save_dir,
        suffix=f"bootstrap_{bootstrap_id:04d}" if save_models else "",
        work_dir=work_dir,
        extra_meta={
            "training_mode": "bootstrap_retrain",
            "bootstrap_id": bootstrap_id,
            "bootstrap_method": bootstrap_sample.method,
            "block_length": bootstrap_sample.block_length,
            "bootstrap_seed": bootstrap_sample.seed,
        },
    )

    row = {
        "dataset": bundle.file_name,
        "task": bundle.task,
        "model": model_name,
        "model_family": "tft",
        "training_mode": "bootstrap_retrain",
        "bootstrap_id": int(bootstrap_id),
        "bootstrap_method": bootstrap_sample.method,
        "block_length": int(bootstrap_sample.block_length),
        "bootstrap_seed": int(bootstrap_sample.seed) if bootstrap_sample.seed is not None else np.nan,
        "bootstrap_unique_fraction": float(bootstrap_sample.unique_fraction),
        "n_dev_original": len(bundle.dev_df),
        "n_dev_bootstrap": len(boot_dev_df),
        "n_test": len(bundle.test_df),
        "trained_this_run": True,
        "model_path": final_results["model_path"] if save_models else "",
    }
    for k, v in final_results.items():
        if k.startswith("test_"):
            row[k] = v
    return row


def run_tft_bootstrap_for_dataset(
    file_name: str,
    n_bootstrap: int = 20,
    bootstrap_method: str = "stationary",
    block_length: Optional[int] = None,
    base_seed: int = 42,
    save_models: bool = False,
    save_dir: str = ROBUSTNESS_DIR,
    result_dir: str = ROBUSTNESS_RESULTS_DIR,
    work_dir: str = ROBUSTNESS_WORK_DIR,
) -> pd.DataFrame:
    ensure_dirs()

    bundle = build_tft_bundle(file_name=file_name)

    if block_length is None:
        block_length = suggest_block_length(len(bundle.dev_df), rule="sqrt", min_block_length=5)

    rows = []

    for model_name, cfg in TFT_MODEL_CONFIGS.items():
        if bundle.task == "classification" and not RUN_TFT_CLASSIFICATION:
            continue

        print(
            f"[TFT BOOTSTRAP] dataset={bundle.file_name} model={model_name} "
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
            row = fit_bootstrap_tft_and_score(
                bundle=bundle,
                model_name=model_name,
                cfg=cfg,
                bootstrap_sample=sample,
                bootstrap_id=b,
                save_models=save_models,
                save_dir=save_dir,
                work_dir=work_dir,
            )
            rows.append(row)

    out_df = pd.DataFrame(rows)
    os.makedirs(result_dir, exist_ok=True)

    out_path = os.path.join(
        result_dir,
        f"tft_bootstrap_{file_name.replace('.csv', '')}_{bootstrap_method}.csv",
    )
    out_df.to_csv(out_path, index=False)
    print(f"Saved TFT bootstrap details to {out_path}")

    summary_df = summarize_tft_bootstrap_results(out_df)
    summary_path = os.path.join(
        result_dir,
        f"tft_bootstrap_summary_{file_name.replace('.csv', '')}_{bootstrap_method}.csv",
    )
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved TFT bootstrap summary to {summary_path}")

    return out_df


def summarize_tft_bootstrap_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    group_cols = ["dataset", "task", "model", "model_family", "training_mode", "bootstrap_method", "block_length"]

    numeric_cols = [
        col for col in df.columns
        if col.startswith("test_") and pd.api.types.is_numeric_dtype(df[col])
    ]

    rows = []
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
            row[f"{col}_ci_low_95"] = float(np.quantile(vals, 0.025))
            row[f"{col}_ci_high_95"] = float(np.quantile(vals, 0.975))

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["dataset", "model"]).reset_index(drop=True)


def run_tft_bootstrap_all_datasets(
    n_bootstrap: int = 20,
    bootstrap_method: str = "stationary",
    block_length: Optional[int] = None,
    base_seed: int = 42,
    save_models: bool = False,
    save_dir: str = ROBUSTNESS_DIR,
    result_dir: str = ROBUSTNESS_RESULTS_DIR,
    work_dir: str = ROBUSTNESS_WORK_DIR,
) -> pd.DataFrame:
    ensure_dirs()

    frames = []
    for file_name in TFT_DATASET_CONFIG:
        df = run_tft_bootstrap_for_dataset(
            file_name=file_name,
            n_bootstrap=n_bootstrap,
            bootstrap_method=bootstrap_method,
            block_length=block_length,
            base_seed=base_seed,
            save_models=save_models,
            save_dir=save_dir,
            result_dir=result_dir,
            work_dir=work_dir,
        )
        frames.append(df)

    final_df = pd.concat(frames, axis=0, ignore_index=True)

    out_path = os.path.join(
        result_dir,
        f"tft_bootstrap_all_{bootstrap_method}.csv",
    )
    final_df.to_csv(out_path, index=False)
    print(f"Saved combined TFT bootstrap details to {out_path}")

    summary_df = summarize_tft_bootstrap_results(final_df)
    summary_path = os.path.join(
        result_dir,
        f"tft_bootstrap_all_summary_{bootstrap_method}.csv",
    )
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved combined TFT bootstrap summary to {summary_path}")

    return final_df


# -------------------------------------------------------------------
# Walk-forward detail
# -------------------------------------------------------------------

def run_tft_walk_forward_detail_for_dataset(
    file_name: str,
    result_dir: str = ROBUSTNESS_RESULTS_DIR,
    work_dir: str = ROBUSTNESS_WORK_DIR,
) -> pd.DataFrame:
    ensure_dirs()

    bundle = build_tft_bundle(file_name=file_name)
    rows = []

    for model_name, cfg in TFT_MODEL_CONFIGS.items():
        if bundle.task == "classification" and not RUN_TFT_CLASSIFICATION:
            continue

        usable_folds = bundle.folds if not RUN_TFT_LIGHT_CV else bundle.folds[-min(5, len(bundle.folds)):]

        for i, fold in enumerate(usable_folds, start=1):
            train_df = bundle.dev_df.iloc[fold.train_idx].reset_index(drop=True)
            valid_df = bundle.dev_df.iloc[fold.valid_idx].reset_index(drop=True)

            metrics = fit_fold_and_score(
                dataset_name=bundle.file_name,
                model_name=model_name,
                cfg=cfg,
                task=bundle.task,
                train_df=train_df,
                valid_df=valid_df,
                target_col=bundle.target_col,
                feature_cols=bundle.feature_cols,
                work_dir=work_dir,
            )

            row = {
                "dataset": bundle.file_name,
                "task": bundle.task,
                "model": model_name,
                "model_family": "tft",
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
            row.update(metrics)
            rows.append(row)

    out_df = pd.DataFrame(rows)
    os.makedirs(result_dir, exist_ok=True)

    out_path = os.path.join(
        result_dir,
        f"tft_walk_forward_detail_{file_name.replace('.csv', '')}.csv",
    )
    out_df.to_csv(out_path, index=False)
    print(f"Saved TFT walk-forward detail to {out_path}")
    return out_df


if __name__ == "__main__":
    df = run_all_tft_datasets(
        force_retrain=False,
        save_dir=MODEL_DIR,
        result_filename=MAIN_RESULTS_FILENAME,
        result_dir=RESULTS_DIR,
        work_dir=WORK_DIR,
    )
    print(df)