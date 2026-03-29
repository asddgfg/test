# regime_analysis.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bootstrap_utils import suggest_block_length
from splitter import train_dev_test_split
from train_eval import (
    DATASET_CONFIG,
    get_model_registry,
    load_dataset as load_classical_dataset,
    load_model as load_classical_model,
    _safe_predict_proba as classical_safe_predict_proba,
)
from deep_train_eval import (
    DEFAULT_SEQ_LEN,
    TRAINING_CONFIG as DEEP_TRAINING_CONFIG,
    apply_feature_scaler,
    create_sequences,
    evaluate_model,
    load_checkpoint as load_deep_checkpoint,
    make_dataloader,
)
from tft_train_eval import (
    TFT_DATASET_CONFIG,
    TFT_MODEL_CONFIGS,
    TEST_HISTORICAL_STRIDE,
    build_covariate_series,
    build_target_series,
    classification_metrics,
    ensure_series_float32,
    evaluate_loaded_artifacts_on_test,
    get_historical_classification_predictions,
    get_historical_regression_predictions,
    load_dataset as load_tft_dataset,
    load_tft_artifacts,
    regression_metrics,
)
from deep_train_eval import DEVICE as DEEP_DEVICE


RESULTS_DIR = "results"
ROBUSTNESS_RESULTS_DIR = os.path.join(RESULTS_DIR, "robustness")

CLASSICAL_MODEL_DIR = "models_saved"
DEEP_MODEL_DIR = "models_saved_deep"
TFT_MODEL_DIR = "models_saved_tft"


@dataclass
class RegimeSlice:
    name: str
    mask: np.ndarray
    n_obs: int


def ensure_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ROBUSTNESS_RESULTS_DIR, exist_ok=True)


# -------------------------------------------------------------------
# Regime construction
# -------------------------------------------------------------------

def rolling_volatility(ret: pd.Series, window: int = 20) -> pd.Series:
    return ret.rolling(window).std()


def compute_basic_market_regimes(
    test_df: pd.DataFrame,
    price_col: str = "spy_close",
    vol_window: int = 20,
    trend_window: int = 50,
) -> pd.DataFrame:
    """
    Build simple market regime labels on the test set.

    Regimes included:
    - high_vol / low_vol
    - bull / bear   (price vs moving average)
    - up_day / down_day
    - positive_return / negative_return

    Notes
    -----
    This is intentionally simple and transparent.
    For the paper, this is often preferable to an over-engineered regime definition.
    """
    df = test_df.copy().reset_index(drop=True)

    if price_col not in df.columns:
        raise ValueError(f"Missing price column '{price_col}' in test_df.")

    px = df[price_col].astype(float)
    ret_1 = np.log(px / px.shift(1))
    vol_20 = rolling_volatility(ret_1, vol_window)
    ma_trend = px.rolling(trend_window).mean()

    vol_median = vol_20.median(skipna=True)

    df["regime_high_vol"] = (vol_20 > vol_median).astype(int)
    df["regime_low_vol"] = (vol_20 <= vol_median).astype(int)

    df["regime_bull"] = (px >= ma_trend).astype(int)
    df["regime_bear"] = (px < ma_trend).astype(int)

    df["regime_up_day"] = (ret_1 > 0).astype(int)
    df["regime_down_day"] = (ret_1 <= 0).astype(int)

    df["ret_1_log"] = ret_1
    df["vol_20"] = vol_20
    df["ma_trend"] = ma_trend

    return df


def available_regime_slices(regime_df: pd.DataFrame) -> List[RegimeSlice]:
    candidates = [
        ("high_vol", regime_df["regime_high_vol"].fillna(0).astype(bool).values),
        ("low_vol", regime_df["regime_low_vol"].fillna(0).astype(bool).values),
        ("bull", regime_df["regime_bull"].fillna(0).astype(bool).values),
        ("bear", regime_df["regime_bear"].fillna(0).astype(bool).values),
        ("up_day", regime_df["regime_up_day"].fillna(0).astype(bool).values),
        ("down_day", regime_df["regime_down_day"].fillna(0).astype(bool).values),
    ]

    out = []
    for name, mask in candidates:
        mask = np.asarray(mask, dtype=bool)
        n_obs = int(mask.sum())
        if n_obs > 0:
            out.append(RegimeSlice(name=name, mask=mask, n_obs=n_obs))
    return out


# -------------------------------------------------------------------
# Generic metric helpers
# -------------------------------------------------------------------

def safe_regression_metrics_subset(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "oos_r2": np.nan,
            "directional_accuracy": np.nan,
        }

    return regression_metrics(y_true, y_pred)


def safe_classification_metrics_subset(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) == 0:
        return {
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "brier": np.nan,
            "log_loss": np.nan,
            "positive_rate_pred": np.nan,
            "positive_rate_true": np.nan,
            "n_positive_true": 0,
            "n_negative_true": 0,
            "roc_auc": np.nan,
        }

    return classification_metrics(y_true, y_prob, threshold=threshold)


def metric_delta_from_overall(
    regime_metrics: Dict[str, float],
    overall_metrics: Dict[str, float],
) -> Dict[str, float]:
    out = {}
    for k, v in regime_metrics.items():
        ov = overall_metrics.get(k, np.nan)
        if pd.isna(v) or pd.isna(ov):
            out[f"{k}_delta_vs_overall"] = np.nan
        else:
            out[f"{k}_delta_vs_overall"] = float(v - ov)
    return out


# -------------------------------------------------------------------
# Classical ML regime analysis
# -------------------------------------------------------------------

def get_classical_test_predictions(
    file_name: str,
    model_name: str,
    model_dir: str = CLASSICAL_MODEL_DIR,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Optional[np.ndarray], str]:
    config = DATASET_CONFIG[file_name]
    target_col = config["target"]
    task = config["task"]

    df = load_classical_dataset(file_name)
    _, test_df = train_dev_test_split(df, test_size=0.2)

    feature_cols = [c for c in test_df.columns if c not in ["Date", target_col]]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].values

    model = load_classical_model(file_name, model_name, save_dir=model_dir)
    y_pred = model.predict(X_test)

    y_prob = None
    if task == "classification":
        y_prob = classical_safe_predict_proba(model, X_test)

    return test_df.reset_index(drop=True), np.asarray(y_test), np.asarray(y_pred), y_prob, task


def run_classical_regime_analysis_for_dataset(
    file_name: str,
    price_col: str = "spy_close",
    model_dir: str = CLASSICAL_MODEL_DIR,
    result_dir: str = ROBUSTNESS_RESULTS_DIR,
) -> pd.DataFrame:
    ensure_dirs()

    rows = []
    task = DATASET_CONFIG[file_name]["task"]

    for model_name in get_model_registry(task).keys():
        print(f"[CLASSICAL REGIME] dataset={file_name} model={model_name}")

        test_df, y_true, y_pred, y_prob, task = get_classical_test_predictions(
            file_name=file_name,
            model_name=model_name,
            model_dir=model_dir,
        )

        regime_df = compute_basic_market_regimes(test_df=test_df, price_col=price_col)
        slices = available_regime_slices(regime_df)

        if task == "regression":
            overall_metrics = safe_regression_metrics_subset(y_true, y_pred)
        else:
            if y_prob is None:
                continue
            overall_metrics = safe_classification_metrics_subset(y_true, y_prob)

        for sl in slices:
            mask = sl.mask
            if task == "regression":
                regime_metrics = safe_regression_metrics_subset(y_true[mask], y_pred[mask])
            else:
                regime_metrics = safe_classification_metrics_subset(y_true[mask], y_prob[mask])

            row = {
                "dataset": file_name,
                "task": task,
                "model": model_name,
                "model_family": "classical_ml",
                "regime_name": sl.name,
                "regime_n_obs": sl.n_obs,
                "test_n_obs": int(len(test_df)),
                "regime_fraction": float(sl.n_obs / len(test_df)) if len(test_df) > 0 else np.nan,
            }
            row.update({f"overall_{k}": v for k, v in overall_metrics.items()})
            row.update({f"regime_{k}": v for k, v in regime_metrics.items()})
            row.update(metric_delta_from_overall(
                {f"regime_{k}": v for k, v in regime_metrics.items()},
                {f"regime_{k}": overall_metrics.get(k, np.nan) for k in regime_metrics.keys()},
            ))
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(
        result_dir,
        f"classical_regime_analysis_{file_name.replace('.csv', '')}.csv",
    )
    out_df.to_csv(out_path, index=False)
    print(f"Saved classical regime analysis to {out_path}")
    return out_df


# -------------------------------------------------------------------
# Deep learning regime analysis
# -------------------------------------------------------------------

def get_deep_test_predictions(
    file_name: str,
    model_name: str,
    seq_len: int = DEFAULT_SEQ_LEN,
    model_dir: str = DEEP_MODEL_DIR,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, str]:
    config = DATASET_CONFIG[file_name]
    target_col = config["target"]
    task = config["task"]

    df = load_classical_dataset(file_name)
    feature_cols = [c for c in df.columns if c not in ["Date", target_col]]

    X_raw = df[feature_cols].values.astype(np.float32)
    y_raw = df[target_col].values.astype(np.float32)
    dates = df["Date"].values

    X_seq, y_seq, seq_dates = create_sequences(X_raw, y_raw, dates, seq_len=seq_len)

    seq_df = pd.DataFrame({
        "Date": pd.to_datetime(seq_dates),
        "seq_idx": np.arange(len(seq_dates)),
    })

    _, test_seq_df = train_dev_test_split(seq_df, test_size=0.2)
    test_idx = test_seq_df["seq_idx"].to_numpy()

    X_test_raw = X_seq[test_idx]
    y_test = y_seq[test_idx]

    model, scaler, checkpoint = load_deep_checkpoint(
        dataset_name=file_name,
        model_name=model_name,
        input_dim=X_test_raw.shape[2],
        seq_len=seq_len,
        save_dir=model_dir,
    )

    X_test = apply_feature_scaler(scaler, X_test_raw)
    loader = make_dataloader(
        X=X_test,
        y=y_test,
        batch_size=DEEP_TRAINING_CONFIG["batch_size"],
        shuffle=False,
    )

    _, metrics = evaluate_model(model, loader, task)

    model.eval()
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEEP_DEVICE)
            logits = model(xb).squeeze(-1)

            if task == "regression":
                outputs = logits.detach().cpu().numpy()
            else:
                outputs = torch.sigmoid(logits).detach().cpu().numpy()

            all_outputs.append(outputs)
            all_targets.append(yb.numpy())

    y_out = np.concatenate(all_outputs)
    y_true = np.concatenate(all_targets)

    # For regime definition alignment, map back to the corresponding original rows
    aligned_test_df = df.iloc[seq_len - 1 :].reset_index(drop=True).iloc[test_idx].reset_index(drop=True)

    return aligned_test_df, y_true, y_out, task


def run_deep_regime_analysis_for_dataset(
    file_name: str,
    deep_model_names: Optional[List[str]] = None,
    seq_len: int = DEFAULT_SEQ_LEN,
    price_col: str = "spy_close",
    model_dir: str = DEEP_MODEL_DIR,
    result_dir: str = ROBUSTNESS_RESULTS_DIR,
) -> pd.DataFrame:
    ensure_dirs()

    from deep_learning_models import get_deep_models

    if deep_model_names is None:
        deep_model_names = list(get_deep_models().keys())

    rows = []

    for model_name in deep_model_names:
        print(f"[DEEP REGIME] dataset={file_name} model={model_name}")

        test_df, y_true, y_out, task = get_deep_test_predictions(
            file_name=file_name,
            model_name=model_name,
            seq_len=seq_len,
            model_dir=model_dir,
        )

        regime_df = compute_basic_market_regimes(test_df=test_df, price_col=price_col)
        slices = available_regime_slices(regime_df)

        if task == "regression":
            overall_metrics = safe_regression_metrics_subset(y_true, y_out)
        else:
            overall_metrics = safe_classification_metrics_subset(y_true, y_out)

        for sl in slices:
            mask = sl.mask

            if task == "regression":
                regime_metrics = safe_regression_metrics_subset(y_true[mask], y_out[mask])
            else:
                regime_metrics = safe_classification_metrics_subset(y_true[mask], y_out[mask])

            row = {
                "dataset": file_name,
                "task": task,
                "model": model_name,
                "model_family": "deep_learning",
                "regime_name": sl.name,
                "regime_n_obs": sl.n_obs,
                "test_n_obs": int(len(test_df)),
                "regime_fraction": float(sl.n_obs / len(test_df)) if len(test_df) > 0 else np.nan,
            }
            row.update({f"overall_{k}": v for k, v in overall_metrics.items()})
            row.update({f"regime_{k}": v for k, v in regime_metrics.items()})
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(
        result_dir,
        f"deep_regime_analysis_{file_name.replace('.csv', '')}.csv",
    )
    out_df.to_csv(out_path, index=False)
    print(f"Saved deep regime analysis to {out_path}")
    return out_df


# -------------------------------------------------------------------
# TFT regime analysis
# -------------------------------------------------------------------

def get_tft_test_predictions(
    file_name: str,
    model_name: str,
    model_dir: str = TFT_MODEL_DIR,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, str]:
    config = TFT_DATASET_CONFIG[file_name]
    target_col = config["target"]
    task = config["task"]

    df = load_tft_dataset(file_name)
    feature_cols = [c for c in df.columns if c not in ["Date", target_col, "time_idx"]]

    dev_df, test_df = train_dev_test_split(df, test_size=0.2)

    artifacts = load_tft_artifacts(
        dataset_name=file_name,
        model_name=model_name,
        expected_task=task,
        expected_target_col=target_col,
        expected_feature_cols=feature_cols,
        expected_cfg=TFT_MODEL_CONFIGS[model_name],
        save_dir=model_dir,
    )

    combined_df = pd.concat([dev_df, test_df], axis=0).reset_index(drop=True)

    if task == "regression":
        combined_target_scaled = ensure_series_float32(
            artifacts.target_scaler.transform(build_target_series(combined_df, target_col))
        )
        combined_cov_scaled = ensure_series_float32(
            artifacts.cov_scaler.transform(build_covariate_series(combined_df, feature_cols))
        )
        combined_target_unscaled = build_target_series(combined_df, target_col)

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
        return test_df.reset_index(drop=True), y_true, y_pred, task

    combined_target = build_target_series(combined_df, target_col)
    combined_cov_scaled = ensure_series_float32(
        artifacts.cov_scaler.transform(build_covariate_series(combined_df, feature_cols))
    )
    start_time = len(dev_df)
    y_true, _, y_prob = get_historical_classification_predictions(
        model=artifacts.model,
        combined_target=combined_target,
        combined_cov_scaled=combined_cov_scaled,
        start_time=start_time,
        stride=TEST_HISTORICAL_STRIDE,
    )
    return test_df.reset_index(drop=True), y_true, y_prob, task


def run_tft_regime_analysis_for_dataset(
    file_name: str,
    price_col: str = "spy_close",
    model_dir: str = TFT_MODEL_DIR,
    result_dir: str = ROBUSTNESS_RESULTS_DIR,
) -> pd.DataFrame:
    ensure_dirs()

    rows = []

    for model_name in TFT_MODEL_CONFIGS.keys():
        if file_name not in TFT_DATASET_CONFIG:
            continue

        print(f"[TFT REGIME] dataset={file_name} model={model_name}")

        test_df, y_true, y_out, task = get_tft_test_predictions(
            file_name=file_name,
            model_name=model_name,
            model_dir=model_dir,
        )

        regime_df = compute_basic_market_regimes(test_df=test_df, price_col=price_col)
        slices = available_regime_slices(regime_df)

        if task == "regression":
            overall_metrics = safe_regression_metrics_subset(y_true, y_out)
        else:
            overall_metrics = safe_classification_metrics_subset(y_true, y_out)

        for sl in slices:
            mask = sl.mask

            if task == "regression":
                regime_metrics = safe_regression_metrics_subset(y_true[mask], y_out[mask])
            else:
                regime_metrics = safe_classification_metrics_subset(y_true[mask], y_out[mask])

            row = {
                "dataset": file_name,
                "task": task,
                "model": model_name,
                "model_family": "tft",
                "regime_name": sl.name,
                "regime_n_obs": sl.n_obs,
                "test_n_obs": int(len(test_df)),
                "regime_fraction": float(sl.n_obs / len(test_df)) if len(test_df) > 0 else np.nan,
            }
            row.update({f"overall_{k}": v for k, v in overall_metrics.items()})
            row.update({f"regime_{k}": v for k, v in regime_metrics.items()})
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(
        result_dir,
        f"tft_regime_analysis_{file_name.replace('.csv', '')}.csv",
    )
    out_df.to_csv(out_path, index=False)
    print(f"Saved TFT regime analysis to {out_path}")
    return out_df


# -------------------------------------------------------------------
# Combined runners
# -------------------------------------------------------------------

def run_classical_regime_analysis_all() -> pd.DataFrame:
    frames = []
    for file_name in DATASET_CONFIG:
        df = run_classical_regime_analysis_for_dataset(file_name=file_name)
        frames.append(df)

    out_df = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
    if not out_df.empty:
        out_path = os.path.join(ROBUSTNESS_RESULTS_DIR, "classical_regime_analysis_all.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Saved combined classical regime analysis to {out_path}")
    return out_df


def run_deep_regime_analysis_all(seq_len: int = DEFAULT_SEQ_LEN) -> pd.DataFrame:
    frames = []
    for file_name in DATASET_CONFIG:
        df = run_deep_regime_analysis_for_dataset(file_name=file_name, seq_len=seq_len)
        frames.append(df)

    out_df = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
    if not out_df.empty:
        out_path = os.path.join(ROBUSTNESS_RESULTS_DIR, "deep_regime_analysis_all.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Saved combined deep regime analysis to {out_path}")
    return out_df


def run_tft_regime_analysis_all() -> pd.DataFrame:
    frames = []
    for file_name in TFT_DATASET_CONFIG:
        df = run_tft_regime_analysis_for_dataset(file_name=file_name)
        frames.append(df)

    out_df = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
    if not out_df.empty:
        out_path = os.path.join(ROBUSTNESS_RESULTS_DIR, "tft_regime_analysis_all.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Saved combined TFT regime analysis to {out_path}")
    return out_df


def run_all_regime_analysis(seq_len: int = DEFAULT_SEQ_LEN) -> pd.DataFrame:
    ensure_dirs()

    classical_df = run_classical_regime_analysis_all()
    deep_df = run_deep_regime_analysis_all(seq_len=seq_len)
    tft_df = run_tft_regime_analysis_all()

    frames = [df for df in [classical_df, deep_df, tft_df] if df is not None and not df.empty]
    final_df = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()

    if not final_df.empty:
        out_path = os.path.join(ROBUSTNESS_RESULTS_DIR, "regime_analysis_all_models.csv")
        final_df.to_csv(out_path, index=False)
        print(f"Saved unified regime analysis to {out_path}")

    return final_df


if __name__ == "__main__":
    df = run_all_regime_analysis(seq_len=DEFAULT_SEQ_LEN)
    print(df.head())