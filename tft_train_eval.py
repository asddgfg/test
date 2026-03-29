import os
import json
import pickle
import shutil
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)

from splitter import expanding_window_splits, train_dev_test_split


warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
MODEL_DIR = "models_saved_tft"
WORK_DIR = os.path.join(MODEL_DIR, "tft_workdir")

# True: also run classification TFT on direction/regime datasets
RUN_TFT_CLASSIFICATION = True

# False: skip light CV and only do final train + test
# True: run a small expanding-window CV before final training
RUN_TFT_LIGHT_CV = True

# If True, remove previous TFT checkpoints/workdir before running
CLEAN_OLD_TFT_ARTIFACTS = False

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


def ensure_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)


def clean_old_artifacts_if_needed() -> None:
    if not CLEAN_OLD_TFT_ARTIFACTS:
        return

    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR, ignore_errors=True)
    os.makedirs(WORK_DIR, exist_ok=True)


def load_dataset(file_name: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing processed dataset: {path}")

    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"Dataset must contain a 'Date' column: {file_name}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

    numeric_cols = [col for col in df.columns if col != "Date"]
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].astype(np.float32)

    return df


def with_local_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["time_idx"] = np.arange(len(out), dtype=np.int64)
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
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

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
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob).astype(float).reshape(-1)
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
        col
        for col in df.columns
        if col not in {"fold", "fold_start", "fold_end"} and pd.api.types.is_numeric_dtype(df[col])
    ]

    for col in metric_cols:
        summary[f"{col}_cv_mean"] = float(df[col].mean())
        summary[f"{col}_cv_std"] = float(df[col].std()) if len(df[col]) > 1 else 0.0

    return summary


def model_path(dataset_name: str, model_name: str) -> str:
    safe_dataset = dataset_name.replace(".csv", "")
    return os.path.join(MODEL_DIR, f"{safe_dataset}__{model_name}.pt")


def meta_path(dataset_name: str, model_name: str) -> str:
    return model_path(dataset_name, model_name) + ".meta.json"


def scaler_path(dataset_name: str, model_name: str) -> str:
    return model_path(dataset_name, model_name) + ".scalers.pkl"


def build_early_stopping(cfg: Dict) -> EarlyStopping:
    patience = 2 if cfg["n_epochs"] <= 8 else 3
    return EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
    )


def get_accelerator() -> str:
    return "gpu" if torch.cuda.is_available() else "cpu"


def get_precision() -> str:
    return "bf16-mixed" if torch.cuda.is_available() else "32-true"


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
            "accelerator": get_accelerator(),
            "devices": 1,
            "precision": get_precision(),
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

    if isinstance(preds_scaled, list):
        preds_scaled = TimeSeries.concatenate(preds_scaled, axis=0)

    preds = target_scaler.inverse_transform(preds_scaled)
    pred_values = preds.values(copy=False).reshape(-1)
    actual_values = combined_target_unscaled.slice_intersect(preds).values(copy=False).reshape(-1)
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

    if isinstance(preds_logits, list):
        preds_logits = TimeSeries.concatenate(preds_logits, axis=0)

    logits = preds_logits.values(copy=False).reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_true = combined_target.slice_intersect(preds_logits).values(copy=False).reshape(-1).astype(int)
    return y_true, logits, probs


def split_dev_for_final_training(
    dev_df: pd.DataFrame,
    final_valid_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(dev_df) <= final_valid_size:
        raise ValueError(
            f"Development set too small for final validation split: "
            f"len(dev_df)={len(dev_df)}, final_valid_size={final_valid_size}"
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
    prepared: PreparedSeries,
) -> Dict[str, str]:
    saved_model_path = model_path(dataset_name, model_name)
    saved_meta_path = meta_path(dataset_name, model_name)
    saved_scaler_path = scaler_path(dataset_name, model_name)

    model.save(saved_model_path)

    scaler_payload = {
        "task": task,
        "target_scaler": prepared.target_scaler,
        "cov_scaler": prepared.cov_scaler,
    }
    with open(saved_scaler_path, "wb") as f:
        pickle.dump(scaler_payload, f)

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
        "model_path": saved_model_path,
        "meta_path": saved_meta_path,
        "scaler_path": saved_scaler_path,
    }

    with open(saved_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return {
        "model_path": saved_model_path,
        "meta_path": saved_meta_path,
        "scaler_path": saved_scaler_path,
    }


def fit_one_model(
    dataset_name: str,
    model_name: str,
    cfg: Dict,
    task: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
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
        work_subdir=WORK_DIR,
    )

    fit_kwargs = {
        "series": prepared.train_target_scaled,
        "future_covariates": prepared.train_cov_scaled,
        "val_series": prepared.valid_target_scaled,
        "val_future_covariates": prepared.valid_cov_scaled,
        "verbose": False,
    }

    dataloader_kwargs = {
        "num_workers": 4 if torch.cuda.is_available() else 0,
        "pin_memory": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available() and dataloader_kwargs["num_workers"] > 0:
        dataloader_kwargs["persistent_workers"] = True

    try:
        model.fit(**fit_kwargs, dataloader_kwargs=dataloader_kwargs)
    except TypeError:
        model.fit(**fit_kwargs)

    return model, prepared


def fit_fold_and_score(
    dataset_name: str,
    model_name: str,
    cfg: Dict,
    task: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
) -> Dict[str, float]:
    fold_model, prepared = fit_one_model(
        dataset_name=dataset_name,
        model_name=f"{dataset_name.replace('.csv', '')}__{model_name}__fold",
        cfg=cfg,
        task=task,
        train_df=train_df,
        valid_df=valid_df,
        target_col=target_col,
        feature_cols=feature_cols,
    )

    combined_df = pd.concat([train_df, valid_df], axis=0, ignore_index=True)
    combined_cov_scaled = ensure_series_float32(
        prepared.cov_scaler.transform(build_covariate_series(combined_df, feature_cols))
    )

    if task == "regression":
        if prepared.target_scaler is None:
            raise ValueError("Regression task requires a fitted target scaler.")

        combined_target_scaled = ensure_series_float32(
            prepared.target_scaler.transform(build_target_series(combined_df, target_col))
        )
        combined_target_unscaled = build_target_series(combined_df, target_col)

        y_true, y_pred = get_historical_regression_predictions(
            model=fold_model,
            target_scaler=prepared.target_scaler,
            combined_target_scaled=combined_target_scaled,
            combined_cov_scaled=combined_cov_scaled,
            combined_target_unscaled=combined_target_unscaled,
            start_time=len(train_df),
            stride=CV_HISTORICAL_STRIDE,
        )
        return regression_metrics(y_true, y_pred)

    combined_target = build_target_series(combined_df, target_col)
    y_true, _, y_prob = get_historical_classification_predictions(
        model=fold_model,
        combined_target=combined_target,
        combined_cov_scaled=combined_cov_scaled,
        start_time=len(train_df),
        stride=CV_HISTORICAL_STRIDE,
    )
    return classification_metrics(y_true, y_prob)


def fit_final_and_test(
    dataset_name: str,
    model_name: str,
    cfg: Dict,
    task: str,
    final_train_df: pd.DataFrame,
    final_valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
) -> Dict[str, float]:
    final_model, prepared = fit_one_model(
        dataset_name=dataset_name,
        model_name=f"{dataset_name.replace('.csv', '')}__{model_name}",
        cfg=cfg,
        task=task,
        train_df=final_train_df,
        valid_df=final_valid_df,
        target_col=target_col,
        feature_cols=feature_cols,
    )

    combined_df = pd.concat([final_train_df, final_valid_df, test_df], axis=0, ignore_index=True)
    combined_cov_scaled = ensure_series_float32(
        prepared.cov_scaler.transform(build_covariate_series(combined_df, feature_cols))
    )

    if task == "regression":
        if prepared.target_scaler is None:
            raise ValueError("Regression task requires a fitted target scaler.")

        combined_target_scaled = ensure_series_float32(
            prepared.target_scaler.transform(build_target_series(combined_df, target_col))
        )
        combined_target_unscaled = build_target_series(combined_df, target_col)

        y_true, y_pred = get_historical_regression_predictions(
            model=final_model,
            target_scaler=prepared.target_scaler,
            combined_target_scaled=combined_target_scaled,
            combined_cov_scaled=combined_cov_scaled,
            combined_target_unscaled=combined_target_unscaled,
            start_time=len(final_train_df) + len(final_valid_df),
            stride=TEST_HISTORICAL_STRIDE,
        )
        test_metrics = regression_metrics(y_true, y_pred)
    else:
        combined_target = build_target_series(combined_df, target_col)
        y_true, _, y_prob = get_historical_classification_predictions(
            model=final_model,
            combined_target=combined_target,
            combined_cov_scaled=combined_cov_scaled,
            start_time=len(final_train_df) + len(final_valid_df),
            stride=TEST_HISTORICAL_STRIDE,
        )
        test_metrics = classification_metrics(y_true, y_prob)

    saved_paths = save_tft_artifacts(
        model=final_model,
        dataset_name=dataset_name,
        model_name=model_name,
        task=task,
        target_col=target_col,
        feature_cols=feature_cols,
        cfg=cfg,
        prepared=prepared,
    )

    return {
        **saved_paths,
        "trained_this_run": True,
        **test_metrics,
    }


def build_output_row(
    file_name: str,
    task: str,
    model_name: str,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    folds: List,
    cv_summary: Dict[str, float],
    test_result: Dict[str, float],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "dataset": file_name,
        "task": task,
        "model": model_name,
        "model_family": "tft",
        "trained_this_run": bool(test_result.get("trained_this_run", True)),
        "exploratory": bool(task == "classification"),
        "light_cv_enabled": RUN_TFT_LIGHT_CV,
        "n_dev": len(dev_df),
        "n_test": len(test_df),
        "n_folds": len(folds),
        **cv_summary,
        "model_path": test_result["model_path"],
    }

    if task == "regression":
        row.update(
            {
                "test_rmse": test_result["rmse"],
                "test_mae": test_result["mae"],
                "test_oos_r2": test_result["oos_r2"],
                "test_directional_accuracy": test_result["directional_accuracy"],
            }
        )
    else:
        row.update(
            {
                "test_accuracy": test_result["accuracy"],
                "test_precision": test_result["precision"],
                "test_recall": test_result["recall"],
                "test_f1": test_result["f1"],
                "test_roc_auc": test_result["roc_auc"],
                "test_brier": test_result["brier"],
                "test_log_loss": test_result["log_loss"],
                "test_positive_rate_pred": test_result["positive_rate_pred"],
                "test_positive_rate_true": test_result["positive_rate_true"],
            }
        )

    return row


def run_tft_cv_for_dataset(
    file_name: str,
    split_mode: str = "expanding",
    test_size: float = DEFAULT_TEST_SIZE,
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
    valid_size: int = DEFAULT_VALID_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
) -> pd.DataFrame:
    ensure_dirs()

    if split_mode != "expanding":
        raise ValueError("This TFT version only supports split_mode='expanding'.")

    if file_name not in TFT_DATASET_CONFIG:
        raise ValueError(f"Unknown dataset config: {file_name}")

    config = TFT_DATASET_CONFIG[file_name]
    target_col = config["target"]
    task = config["task"]
    purge = config["purge"]
    embargo = config["embargo"]

    if task == "classification" and not RUN_TFT_CLASSIFICATION:
        return pd.DataFrame()

    df = load_dataset(file_name)
    feature_cols = get_feature_columns(df, target_col)

    dev_df, test_df = train_dev_test_split(df, test_size=test_size)
    final_train_df, final_valid_df = split_dev_for_final_training(
        dev_df=dev_df,
        final_valid_size=FINAL_VALID_SIZE,
    )

    folds: List = []
    if RUN_TFT_LIGHT_CV:
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
        if len(folds) > 3:
            folds = folds[-3:]

    all_results: List[Dict[str, object]] = []

    for model_name, cfg in TFT_MODEL_CONFIGS.items():
        print(f"[TFT] {file_name} / {model_name} / task={task}")
        fold_records: List[Dict[str, float]] = []

        if RUN_TFT_LIGHT_CV:
            for i, fold in enumerate(folds, start=1):
                train_df = dev_df.iloc[fold.train_idx].reset_index(drop=True)
                valid_df = dev_df.iloc[fold.valid_idx].reset_index(drop=True)

                metrics = fit_fold_and_score(
                    dataset_name=file_name,
                    model_name=model_name,
                    cfg=cfg,
                    task=task,
                    train_df=train_df,
                    valid_df=valid_df,
                    target_col=target_col,
                    feature_cols=feature_cols,
                )
                metrics["fold"] = i
                metrics["fold_start"] = int(fold.valid_idx[0])
                metrics["fold_end"] = int(fold.valid_idx[-1])
                fold_records.append(metrics)

        cv_summary = summarize_fold_metrics(fold_records)

        test_result = fit_final_and_test(
            dataset_name=file_name,
            model_name=model_name,
            cfg=cfg,
            task=task,
            final_train_df=final_train_df,
            final_valid_df=final_valid_df,
            test_df=test_df,
            target_col=target_col,
            feature_cols=feature_cols,
        )

        row = build_output_row(
            file_name=file_name,
            task=task,
            model_name=model_name,
            dev_df=dev_df,
            test_df=test_df,
            folds=folds,
            cv_summary=cv_summary,
            test_result=test_result,
        )
        all_results.append(row)

    return pd.DataFrame(all_results)


def run_all_tft_datasets(split_mode: str = "expanding") -> pd.DataFrame:
    ensure_dirs()
    clean_old_artifacts_if_needed()

    result_frames: List[pd.DataFrame] = []
    exploratory_frames: List[pd.DataFrame] = []

    for file_name, cfg in TFT_DATASET_CONFIG.items():
        result_df = run_tft_cv_for_dataset(file_name, split_mode=split_mode)
        if result_df.empty:
            continue

        if cfg["task"] == "classification":
            exploratory_frames.append(result_df)
        else:
            result_frames.append(result_df)

    main_df = pd.concat(result_frames, axis=0, ignore_index=True) if result_frames else pd.DataFrame()
    exploratory_df = (
        pd.concat(exploratory_frames, axis=0, ignore_index=True) if exploratory_frames else pd.DataFrame()
    )

    main_out_path = os.path.join(RESULTS_DIR, MAIN_RESULTS_FILENAME)
    main_df.to_csv(main_out_path, index=False)
    print(f"Saved main TFT results to {main_out_path}")

    if RUN_TFT_CLASSIFICATION:
        exploratory_out_path = os.path.join(RESULTS_DIR, EXPLORATORY_RESULTS_FILENAME)
        exploratory_df.to_csv(exploratory_out_path, index=False)
        print(f"Saved exploratory TFT classification results to {exploratory_out_path}")

    return main_df


if __name__ == "__main__":
    df = run_all_tft_datasets(split_mode="expanding")
    print(df)