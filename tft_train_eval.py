import os
import json
import pickle
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from sklearn.linear_model import LogisticRegression
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


PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
MODEL_DIR = "models_saved_tft"

# Now supports regression + binary classification
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
        "hidden_size": 16,
        "lstm_layers": 1,
        "num_attention_heads": 4,
        "dropout": 0.1,
        "hidden_continuous_size": 8,
        "batch_size": 32,
        "n_epochs": 20,
        "learning_rate": 1e-3,
    },
    "tft_medium": {
        "input_chunk_length": 30,
        "output_chunk_length": 1,
        "hidden_size": 32,
        "lstm_layers": 1,
        "num_attention_heads": 4,
        "dropout": 0.1,
        "hidden_continuous_size": 16,
        "batch_size": 32,
        "n_epochs": 25,
        "learning_rate": 1e-3,
    },
    "tft_large": {
        "input_chunk_length": 40,
        "output_chunk_length": 1,
        "hidden_size": 32,
        "lstm_layers": 2,
        "num_attention_heads": 4,
        "dropout": 0.2,
        "hidden_continuous_size": 16,
        "batch_size": 32,
        "n_epochs": 30,
        "learning_rate": 5e-4,
    },
}

FINAL_VALID_SIZE = 63
CLASSIFICATION_THRESHOLD = 0.5


def ensure_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def load_dataset(file_name: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing processed dataset: {path}")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def with_local_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["time_idx"] = np.arange(len(out), dtype=int)
    return out


def get_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    return [col for col in df.columns if col not in ["Date", "time_idx", target_col]]


def build_target_series(df: pd.DataFrame, target_col: str) -> TimeSeries:
    local_df = with_local_time_idx(df)
    return TimeSeries.from_dataframe(
        local_df,
        time_col="time_idx",
        value_cols=[target_col],
        fill_missing_dates=False,
    )


def build_covariate_series(df: pd.DataFrame, feature_cols: List[str]) -> TimeSeries:
    local_df = with_local_time_idx(df)
    return TimeSeries.from_dataframe(
        local_df,
        time_col="time_idx",
        value_cols=feature_cols,
        fill_missing_dates=False,
    )


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
    y_prob: np.ndarray,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "positive_rate_pred": float(np.mean(y_pred)),
    }

    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = np.nan

    return metrics


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


def model_path(dataset_name: str, model_name: str) -> str:
    safe_dataset = dataset_name.replace(".csv", "")
    return os.path.join(MODEL_DIR, f"{safe_dataset}__{model_name}.pt")


def calibrator_path(dataset_name: str, model_name: str) -> str:
    safe_dataset = dataset_name.replace(".csv", "")
    return os.path.join(MODEL_DIR, f"{safe_dataset}__{model_name}__platt.pkl")


def meta_path(dataset_name: str, model_name: str) -> str:
    return model_path(dataset_name, model_name) + ".meta.json"


def build_tft_model(model_name: str, cfg: Dict, work_subdir: str) -> TFTModel:
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
        loss_fn=torch.nn.MSELoss(),
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        optimizer_kwargs={"lr": cfg["learning_rate"]},
        random_state=42,
        model_name=model_name,
        work_dir=work_subdir,
        save_checkpoints=False,
        force_reset=True,
        log_tensorboard=False,
        pl_trainer_kwargs={
            "enable_progress_bar": True,
            "logger": False,
        },
    )


def fit_scalers_and_build_series(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    target_col: str,
    feature_cols: List[str],
):
    train_target = build_target_series(train_df, target_col)
    train_cov = build_covariate_series(train_df, feature_cols)

    valid_target = build_target_series(valid_df, target_col)
    valid_cov = build_covariate_series(valid_df, feature_cols)

    target_scaler = Scaler()
    cov_scaler = Scaler()

    train_target_scaled = target_scaler.fit_transform(train_target)
    train_cov_scaled = cov_scaler.fit_transform(train_cov)

    valid_target_scaled = target_scaler.transform(valid_target)
    valid_cov_scaled = cov_scaler.transform(valid_cov)

    test_target_scaled = None
    test_cov_scaled = None

    if test_df is not None:
        test_target = build_target_series(test_df, target_col)
        test_cov = build_covariate_series(test_df, feature_cols)
        test_target_scaled = target_scaler.transform(test_target)
        test_cov_scaled = cov_scaler.transform(test_cov)

    return (
        train_target_scaled,
        train_cov_scaled,
        valid_target_scaled,
        valid_cov_scaled,
        test_target_scaled,
        test_cov_scaled,
        target_scaler,
        cov_scaler,
    )


def get_historical_raw_predictions(
    model: TFTModel,
    target_scaler: Scaler,
    combined_target_scaled: TimeSeries,
    combined_cov_scaled: TimeSeries,
    combined_target_unscaled: TimeSeries,
    start_time: int,
) -> Tuple[np.ndarray, np.ndarray]:
    preds_scaled = model.historical_forecasts(
        series=combined_target_scaled,
        future_covariates=combined_cov_scaled,
        start=start_time,
        forecast_horizon=1,
        stride=1,
        retrain=False,
        last_points_only=True,
        verbose=False,
    )

    preds = target_scaler.inverse_transform(preds_scaled)
    pred_values = preds.values(copy=False).flatten()

    actual = combined_target_unscaled.slice_intersect(preds)
    actual_values = actual.values(copy=False).flatten()

    return actual_values, pred_values


def historical_forecast_metrics_regression(
    model: TFTModel,
    target_scaler: Scaler,
    combined_target_scaled: TimeSeries,
    combined_cov_scaled: TimeSeries,
    combined_target_unscaled: TimeSeries,
    start_time: int,
) -> Dict[str, float]:
    actual_values, pred_values = get_historical_raw_predictions(
        model=model,
        target_scaler=target_scaler,
        combined_target_scaled=combined_target_scaled,
        combined_cov_scaled=combined_cov_scaled,
        combined_target_unscaled=combined_target_unscaled,
        start_time=start_time,
    )
    return regression_metrics(actual_values, pred_values)


def fit_platt_scaler(
    y_true: np.ndarray,
    raw_scores: np.ndarray,
) -> LogisticRegression:
    y_true = np.asarray(y_true).astype(int)
    raw_scores = np.asarray(raw_scores).reshape(-1, 1)

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        raise ValueError(
            "Platt scaling requires both classes in the calibration set. "
            f"Got classes: {unique_classes.tolist()}"
        )

    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(raw_scores, y_true)
    return calibrator


def apply_platt_scaler(
    calibrator: LogisticRegression,
    raw_scores: np.ndarray,
) -> np.ndarray:
    raw_scores = np.asarray(raw_scores).reshape(-1, 1)
    probs = calibrator.predict_proba(raw_scores)[:, 1]
    return probs


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
    calibrator: Optional[LogisticRegression] = None,
) -> Dict[str, Optional[str]]:
    saved_model_path = model_path(dataset_name, model_name)
    model.save(saved_model_path)

    saved_calibrator_path = None
    if calibrator is not None:
        saved_calibrator_path = calibrator_path(dataset_name, model_name)
        with open(saved_calibrator_path, "wb") as f:
            pickle.dump(calibrator, f)

    metadata = {
        "dataset": dataset_name,
        "model": model_name,
        "task": task,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "config": cfg,
        "calibrator_path": saved_calibrator_path,
        "classification_threshold": CLASSIFICATION_THRESHOLD if task == "classification" else None,
    }

    with open(meta_path(dataset_name, model_name), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return {
        "model_path": saved_model_path,
        "calibrator_path": saved_calibrator_path,
    }


def fit_and_score_fold_regression(
    dataset_name: str,
    model_name: str,
    cfg: Dict,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
) -> Dict[str, float]:
    (
        train_target_scaled,
        train_cov_scaled,
        valid_target_scaled,
        valid_cov_scaled,
        _,
        _,
        target_scaler,
        cov_scaler,
    ) = fit_scalers_and_build_series(
        train_df=train_df,
        valid_df=valid_df,
        test_df=None,
        target_col=target_col,
        feature_cols=feature_cols,
    )

    combined_df = pd.concat([train_df, valid_df], axis=0, ignore_index=True)

    combined_target_scaled = target_scaler.transform(
        build_target_series(combined_df, target_col)
    )
    combined_cov_scaled = cov_scaler.transform(
        build_covariate_series(combined_df, feature_cols)
    )
    combined_unscaled_target = build_target_series(combined_df, target_col)

    work_subdir = os.path.join(MODEL_DIR, "tft_workdir")
    os.makedirs(work_subdir, exist_ok=True)

    fold_model = build_tft_model(
        model_name=f"{dataset_name.replace('.csv', '')}__{model_name}__fold",
        cfg=cfg,
        work_subdir=work_subdir,
    )

    fold_model.fit(
        series=train_target_scaled,
        future_covariates=train_cov_scaled,
        val_series=valid_target_scaled,
        val_future_covariates=valid_cov_scaled,
        verbose=False,
    )

    return historical_forecast_metrics_regression(
        model=fold_model,
        target_scaler=target_scaler,
        combined_target_scaled=combined_target_scaled,
        combined_cov_scaled=combined_cov_scaled,
        combined_target_unscaled=combined_unscaled_target,
        start_time=len(train_df),
    )


def fit_and_score_fold_classification(
    dataset_name: str,
    model_name: str,
    cfg: Dict,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
) -> Dict[str, float]:
    (
        train_target_scaled,
        train_cov_scaled,
        valid_target_scaled,
        valid_cov_scaled,
        _,
        _,
        target_scaler,
        cov_scaler,
    ) = fit_scalers_and_build_series(
        train_df=train_df,
        valid_df=valid_df,
        test_df=None,
        target_col=target_col,
        feature_cols=feature_cols,
    )

    combined_df = pd.concat([train_df, valid_df], axis=0, ignore_index=True)

    combined_target_scaled = target_scaler.transform(
        build_target_series(combined_df, target_col)
    )
    combined_cov_scaled = cov_scaler.transform(
        build_covariate_series(combined_df, feature_cols)
    )
    combined_unscaled_target = build_target_series(combined_df, target_col)

    work_subdir = os.path.join(MODEL_DIR, "tft_workdir")
    os.makedirs(work_subdir, exist_ok=True)

    fold_model = build_tft_model(
        model_name=f"{dataset_name.replace('.csv', '')}__{model_name}__fold",
        cfg=cfg,
        work_subdir=work_subdir,
    )

    fold_model.fit(
        series=train_target_scaled,
        future_covariates=train_cov_scaled,
        val_series=valid_target_scaled,
        val_future_covariates=valid_cov_scaled,
        verbose=False,
    )

    y_valid_true, y_valid_raw = get_historical_raw_predictions(
        model=fold_model,
        target_scaler=target_scaler,
        combined_target_scaled=combined_target_scaled,
        combined_cov_scaled=combined_cov_scaled,
        combined_target_unscaled=combined_unscaled_target,
        start_time=len(train_df),
    )

    calibrator = fit_platt_scaler(y_valid_true, y_valid_raw)
    y_valid_prob = apply_platt_scaler(calibrator, y_valid_raw)

    return classification_metrics(y_valid_true, y_valid_prob)


def fit_final_and_test_regression(
    dataset_name: str,
    model_name: str,
    cfg: Dict,
    final_train_df: pd.DataFrame,
    final_valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
) -> Dict[str, float]:
    (
        final_train_target_scaled,
        final_train_cov_scaled,
        final_valid_target_scaled,
        final_valid_cov_scaled,
        _,
        _,
        target_scaler,
        cov_scaler,
    ) = fit_scalers_and_build_series(
        train_df=final_train_df,
        valid_df=final_valid_df,
        test_df=None,
        target_col=target_col,
        feature_cols=feature_cols,
    )

    combined_df = pd.concat(
        [final_train_df, final_valid_df, test_df],
        axis=0,
        ignore_index=True,
    )

    combined_target_scaled = target_scaler.transform(
        build_target_series(combined_df, target_col)
    )
    combined_cov_scaled = cov_scaler.transform(
        build_covariate_series(combined_df, feature_cols)
    )
    combined_unscaled_target = build_target_series(combined_df, target_col)

    work_subdir = os.path.join(MODEL_DIR, "tft_workdir")
    os.makedirs(work_subdir, exist_ok=True)

    final_model = build_tft_model(
        model_name=f"{dataset_name.replace('.csv', '')}__{model_name}",
        cfg=cfg,
        work_subdir=work_subdir,
    )

    final_model.fit(
        series=final_train_target_scaled,
        future_covariates=final_train_cov_scaled,
        val_series=final_valid_target_scaled,
        val_future_covariates=final_valid_cov_scaled,
        verbose=False,
    )

    test_metrics = historical_forecast_metrics_regression(
        model=final_model,
        target_scaler=target_scaler,
        combined_target_scaled=combined_target_scaled,
        combined_cov_scaled=combined_cov_scaled,
        combined_target_unscaled=combined_unscaled_target,
        start_time=len(final_train_df) + len(final_valid_df),
    )

    saved_paths = save_tft_artifacts(
        model=final_model,
        dataset_name=dataset_name,
        model_name=model_name,
        task="regression",
        target_col=target_col,
        feature_cols=feature_cols,
        cfg=cfg,
        calibrator=None,
    )

    return {
        **saved_paths,
        **test_metrics,
    }


def fit_final_and_test_classification(
    dataset_name: str,
    model_name: str,
    cfg: Dict,
    final_train_df: pd.DataFrame,
    final_valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
) -> Dict[str, float]:
    (
        final_train_target_scaled,
        final_train_cov_scaled,
        final_valid_target_scaled,
        final_valid_cov_scaled,
        _,
        _,
        target_scaler,
        cov_scaler,
    ) = fit_scalers_and_build_series(
        train_df=final_train_df,
        valid_df=final_valid_df,
        test_df=None,
        target_col=target_col,
        feature_cols=feature_cols,
    )

    work_subdir = os.path.join(MODEL_DIR, "tft_workdir")
    os.makedirs(work_subdir, exist_ok=True)

    final_model = build_tft_model(
        model_name=f"{dataset_name.replace('.csv', '')}__{model_name}",
        cfg=cfg,
        work_subdir=work_subdir,
    )

    final_model.fit(
        series=final_train_target_scaled,
        future_covariates=final_train_cov_scaled,
        val_series=final_valid_target_scaled,
        val_future_covariates=final_valid_cov_scaled,
        verbose=False,
    )

    # 1) use validation set to fit Platt scaler
    valid_combined_df = pd.concat(
        [final_train_df, final_valid_df],
        axis=0,
        ignore_index=True,
    )
    valid_combined_target_scaled = target_scaler.transform(
        build_target_series(valid_combined_df, target_col)
    )
    valid_combined_cov_scaled = cov_scaler.transform(
        build_covariate_series(valid_combined_df, feature_cols)
    )
    valid_combined_target_unscaled = build_target_series(valid_combined_df, target_col)

    y_valid_true, y_valid_raw = get_historical_raw_predictions(
        model=final_model,
        target_scaler=target_scaler,
        combined_target_scaled=valid_combined_target_scaled,
        combined_cov_scaled=valid_combined_cov_scaled,
        combined_target_unscaled=valid_combined_target_unscaled,
        start_time=len(final_train_df),
    )

    calibrator = fit_platt_scaler(y_valid_true, y_valid_raw)

    # 2) evaluate on test
    test_combined_df = pd.concat(
        [final_train_df, final_valid_df, test_df],
        axis=0,
        ignore_index=True,
    )
    test_combined_target_scaled = target_scaler.transform(
        build_target_series(test_combined_df, target_col)
    )
    test_combined_cov_scaled = cov_scaler.transform(
        build_covariate_series(test_combined_df, feature_cols)
    )
    test_combined_target_unscaled = build_target_series(test_combined_df, target_col)

    y_test_true, y_test_raw = get_historical_raw_predictions(
        model=final_model,
        target_scaler=target_scaler,
        combined_target_scaled=test_combined_target_scaled,
        combined_cov_scaled=test_combined_cov_scaled,
        combined_target_unscaled=test_combined_target_unscaled,
        start_time=len(final_train_df) + len(final_valid_df),
    )

    y_test_prob = apply_platt_scaler(calibrator, y_test_raw)
    test_metrics = classification_metrics(y_test_true, y_test_prob)

    saved_paths = save_tft_artifacts(
        model=final_model,
        dataset_name=dataset_name,
        model_name=model_name,
        task="classification",
        target_col=target_col,
        feature_cols=feature_cols,
        cfg=cfg,
        calibrator=calibrator,
    )

    return {
        **saved_paths,
        **test_metrics,
    }


def run_tft_cv_for_dataset(
    file_name: str,
    split_mode: str = "expanding",
    test_size: float = 0.2,
    min_train_size: int = 504,
    valid_size: int = 63,
    step_size: int = 21,
) -> pd.DataFrame:
    ensure_dirs()

    if split_mode != "expanding":
        raise ValueError("This TFT version only supports split_mode='expanding'.")

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
            f"No valid expanding-window folds for {file_name}. "
            f"Check dataset size, min_train_size, valid_size, step_size."
        )

    final_train_df, final_valid_df = split_dev_for_final_training(
        dev_df=dev_df,
        final_valid_size=FINAL_VALID_SIZE,
    )

    all_results = []

    for model_name, cfg in TFT_MODEL_CONFIGS.items():
        print(f"[TFT] {file_name} / {model_name} / task={task}")
        fold_records = []

        for i, fold in enumerate(folds, start=1):
            train_df = dev_df.iloc[fold.train_idx].reset_index(drop=True)
            valid_df = dev_df.iloc[fold.valid_idx].reset_index(drop=True)

            if task == "regression":
                metrics = fit_and_score_fold_regression(
                    dataset_name=file_name,
                    model_name=model_name,
                    cfg=cfg,
                    train_df=train_df,
                    valid_df=valid_df,
                    target_col=target_col,
                    feature_cols=feature_cols,
                )
            elif task == "classification":
                metrics = fit_and_score_fold_classification(
                    dataset_name=file_name,
                    model_name=model_name,
                    cfg=cfg,
                    train_df=train_df,
                    valid_df=valid_df,
                    target_col=target_col,
                    feature_cols=feature_cols,
                )
            else:
                raise ValueError(f"Unsupported task: {task}")

            metrics["fold"] = i
            fold_records.append(metrics)

        cv_summary = summarize_fold_metrics(fold_records)

        if task == "regression":
            test_result = fit_final_and_test_regression(
                dataset_name=file_name,
                model_name=model_name,
                cfg=cfg,
                final_train_df=final_train_df,
                final_valid_df=final_valid_df,
                test_df=test_df,
                target_col=target_col,
                feature_cols=feature_cols,
            )

            row = {
                "dataset": file_name,
                "task": task,
                "model": model_name,
                "model_family": "tft",
                "trained_this_run": True,
                "n_dev": len(dev_df),
                "n_test": len(test_df),
                "n_folds": len(folds),
                **cv_summary,
                "model_path": test_result["model_path"],
                "calibrator_path": test_result["calibrator_path"],
                "test_rmse": test_result["rmse"],
                "test_mae": test_result["mae"],
                "test_oos_r2": test_result["oos_r2"],
                "test_directional_accuracy": test_result["directional_accuracy"],
            }

        elif task == "classification":
            test_result = fit_final_and_test_classification(
                dataset_name=file_name,
                model_name=model_name,
                cfg=cfg,
                final_train_df=final_train_df,
                final_valid_df=final_valid_df,
                test_df=test_df,
                target_col=target_col,
                feature_cols=feature_cols,
            )

            row = {
                "dataset": file_name,
                "task": task,
                "model": model_name,
                "model_family": "tft",
                "trained_this_run": True,
                "n_dev": len(dev_df),
                "n_test": len(test_df),
                "n_folds": len(folds),
                **cv_summary,
                "model_path": test_result["model_path"],
                "calibrator_path": test_result["calibrator_path"],
                "test_accuracy": test_result["accuracy"],
                "test_precision": test_result["precision"],
                "test_recall": test_result["recall"],
                "test_f1": test_result["f1"],
                "test_roc_auc": test_result["roc_auc"],
                "test_brier": test_result["brier"],
                "test_log_loss": test_result["log_loss"],
                "test_positive_rate_pred": test_result["positive_rate_pred"],
            }

        else:
            raise ValueError(f"Unsupported task: {task}")

        all_results.append(row)

    return pd.DataFrame(all_results)


def run_all_tft_datasets(split_mode: str = "expanding") -> pd.DataFrame:
    ensure_dirs()

    result_frames = []
    for file_name in TFT_DATASET_CONFIG:
        result_df = run_tft_cv_for_dataset(file_name, split_mode=split_mode)
        result_frames.append(result_df)

    final_df = pd.concat(result_frames, axis=0, ignore_index=True)
    out_path = os.path.join(RESULTS_DIR, f"tft_model_comparison_{split_mode}.csv")
    final_df.to_csv(out_path, index=False)

    print(f"Saved TFT results to {out_path}")
    return final_df


if __name__ == "__main__":
    df = run_all_tft_datasets(split_mode="expanding")
    print(df)