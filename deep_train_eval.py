# deep_train_eval.py
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from bootstrap_utils import BootstrapSample, iter_bootstrap_samples, suggest_block_length
from deep_learning_models import build_model, get_deep_models
from splitter import expanding_window_splits, train_dev_test_split
from train_eval import DATASET_CONFIG


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
MODEL_DIR = "models_saved_deep"

ROBUSTNESS_DIR = os.path.join(MODEL_DIR, "robustness")
ROBUSTNESS_RESULTS_DIR = os.path.join(RESULTS_DIR, "robustness")

DEFAULT_SEQ_LEN = 20
DEFAULT_TEST_SIZE = 0.2
DEFAULT_MIN_TRAIN_SIZE = 504
DEFAULT_VALID_SIZE = 63
DEFAULT_STEP_SIZE = 21
DEFAULT_SKIP_EXISTING = True
FINAL_VALID_SIZE = DEFAULT_VALID_SIZE

TRAINING_CONFIG = {
    "epochs": 30,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "patience": 5,
    "min_delta": 1e-4,
    "grad_clip_norm": 1.0,
}


@dataclass
class SequenceBundle:
    file_name: str
    target_col: str
    task: str
    feature_cols: List[str]
    seq_len: int
    seq_df: pd.DataFrame
    X_dev_raw: np.ndarray
    y_dev: np.ndarray
    X_test_raw: np.ndarray
    y_test: np.ndarray
    folds: List
    n_features: int


def ensure_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(ROBUSTNESS_DIR, exist_ok=True)
    os.makedirs(ROBUSTNESS_RESULTS_DIR, exist_ok=True)


def load_dataset(file_name: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing processed dataset: {path}")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def get_target_and_task(file_name: str) -> Tuple[str, str]:
    config = DATASET_CONFIG[file_name]
    return config["target"], config["task"]


def get_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    return [col for col in df.columns if col not in ["Date", target_col]]


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_seq = []
    y_seq = []
    date_seq = []

    for i in range(seq_len - 1, len(X)):
        X_seq.append(X[i - seq_len + 1 : i + 1])
        y_seq.append(y[i])
        date_seq.append(dates[i])

    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32),
        np.asarray(date_seq),
    )


def fit_feature_scaler(X_train_raw: np.ndarray) -> StandardScaler:
    _, _, n_feat = X_train_raw.shape
    scaler = StandardScaler()
    scaler.fit(X_train_raw.reshape(-1, n_feat))
    return scaler


def apply_feature_scaler(
    scaler: StandardScaler,
    X_raw: np.ndarray,
) -> np.ndarray:
    n_obs, seq_len, n_feat = X_raw.shape
    X_scaled = scaler.transform(X_raw.reshape(-1, n_feat)).reshape(n_obs, seq_len, n_feat)
    return X_scaled.astype(np.float32)


def scale_datasets(
    X_train_raw: np.ndarray,
    X_valid_raw: np.ndarray,
    X_test_raw: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], StandardScaler]:
    scaler = fit_feature_scaler(X_train_raw)
    X_train = apply_feature_scaler(scaler, X_train_raw)
    X_valid = apply_feature_scaler(scaler, X_valid_raw)

    X_test = None
    if X_test_raw is not None:
        X_test = apply_feature_scaler(scaler, X_test_raw)

    return X_train, X_valid, X_test, scaler


def make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    positives = float(np.sum(y_train == 1))
    negatives = float(np.sum(y_train == 0))

    if positives <= 0 or negatives <= 0:
        return torch.tensor(1.0, dtype=torch.float32, device=DEVICE)

    ratio = negatives / positives
    return torch.tensor(max(ratio, 1e-6), dtype=torch.float32, device=DEVICE)


def get_train_loss_fn(task: str, y_train: np.ndarray):
    if task == "regression":
        return torch.nn.MSELoss()

    pos_weight = compute_pos_weight(y_train)
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def get_eval_loss_fn(task: str):
    if task == "regression":
        return torch.nn.MSELoss()
    return torch.nn.BCEWithLogitsLoss()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    directional_accuracy = float(np.mean(np.sign(y_true) == np.sign(y_pred)))

    return {
        "rmse": rmse,
        "mae": mae,
        "oos_r2": r2,
        "directional_accuracy": directional_accuracy,
    }


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= 0.5).astype(int)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "positive_rate_pred": float(np.mean(y_pred)),
    }

    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["roc_auc"] = np.nan

    return out


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    task: str,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    eval_loss_fn = get_eval_loss_fn(task)

    all_targets = []
    all_outputs = []
    losses = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(xb).squeeze(-1)
            loss = eval_loss_fn(logits, yb)
            losses.append(float(loss.item()))

            if task == "regression":
                outputs = logits.detach().cpu().numpy()
            else:
                outputs = torch.sigmoid(logits).detach().cpu().numpy()

            all_outputs.append(outputs)
            all_targets.append(yb.detach().cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_out = np.concatenate(all_outputs)

    if task == "regression":
        metrics = regression_metrics(y_true, y_out)
    else:
        metrics = classification_metrics(y_true, y_out)

    avg_loss = float(np.mean(losses)) if losses else np.inf
    return avg_loss, metrics


def train_one_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    task: str,
    y_train: np.ndarray,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
    grad_clip_norm: float,
) -> Tuple[torch.nn.Module, Dict[str, float], float]:
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    train_loss_fn = get_train_loss_fn(task, y_train)

    best_state = copy.deepcopy(model.state_dict())
    best_valid_loss = np.inf
    patience_counter = 0

    for _ in range(epochs):
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = train_loss_fn(logits, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        valid_loss, _ = evaluate_model(model, valid_loader, task)

        if valid_loss < best_valid_loss - min_delta:
            best_valid_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    final_valid_loss, final_valid_metrics = evaluate_model(model, valid_loader, task)
    return model, final_valid_metrics, final_valid_loss


def summarize_cv(records: List[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        return {}

    df = pd.DataFrame(records)
    out = {}

    for col in df.columns:
        if col not in {"fold", "fold_start", "fold_end"} and pd.api.types.is_numeric_dtype(df[col]):
            out[f"{col}_cv_mean"] = float(df[col].mean())
            out[f"{col}_cv_std"] = float(df[col].std()) if len(df[col]) > 1 else 0.0

    return out


def split_dev_for_final_training(
    X_dev_raw: np.ndarray,
    y_dev: np.ndarray,
    final_valid_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(X_dev_raw) <= final_valid_size:
        raise ValueError(
            f"Development set too small for final validation split: "
            f"len(X_dev_raw)={len(X_dev_raw)}, final_valid_size={final_valid_size}"
        )

    split_idx = len(X_dev_raw) - final_valid_size

    X_final_train_raw = X_dev_raw[:split_idx]
    y_final_train = y_dev[:split_idx]

    X_final_valid_raw = X_dev_raw[split_idx:]
    y_final_valid = y_dev[split_idx:]

    return X_final_train_raw, y_final_train, X_final_valid_raw, y_final_valid


def model_checkpoint_path(
    dataset_name: str,
    model_name: str,
    save_dir: str = MODEL_DIR,
    suffix: str = "",
) -> str:
    safe_dataset = dataset_name.replace(".csv", "")
    safe_suffix = f"__{suffix}" if suffix else ""
    return os.path.join(save_dir, f"{safe_dataset}__{model_name}{safe_suffix}.pt")


def model_meta_path(
    dataset_name: str,
    model_name: str,
    save_dir: str = MODEL_DIR,
    suffix: str = "",
) -> str:
    return model_checkpoint_path(dataset_name, model_name, save_dir=save_dir, suffix=suffix) + ".meta.json"


def save_checkpoint(
    model: torch.nn.Module,
    dataset_name: str,
    model_name: str,
    task: str,
    seq_len: int,
    feature_cols: List[str],
    scaler: StandardScaler,
    model_type: str,
    model_params: Dict,
    train_config: Dict,
    save_dir: str = MODEL_DIR,
    suffix: str = "",
    extra_meta: Optional[Dict] = None,
) -> str:
    os.makedirs(save_dir, exist_ok=True)

    path = model_checkpoint_path(dataset_name, model_name, save_dir=save_dir, suffix=suffix)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "task": task,
        "seq_len": seq_len,
        "feature_cols": feature_cols,
        "scaler_mean_": scaler.mean_,
        "scaler_scale_": scaler.scale_,
        "model_type": model_type,
        "model_params": model_params,
        "device_used": str(DEVICE),
    }
    torch.save(checkpoint, path)

    meta = {
        "dataset": dataset_name,
        "model": model_name,
        "task": task,
        "seq_len": seq_len,
        "feature_cols": feature_cols,
        "model_type": model_type,
        "model_params": model_params,
        "train_config": train_config,
    }
    if extra_meta:
        meta.update(extra_meta)

    with open(model_meta_path(dataset_name, model_name, save_dir=save_dir, suffix=suffix), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return path


def load_checkpoint(
    dataset_name: str,
    model_name: str,
    input_dim: int,
    seq_len: int,
    save_dir: str = MODEL_DIR,
    suffix: str = "",
):
    checkpoint_path = model_checkpoint_path(dataset_name, model_name, save_dir=save_dir, suffix=suffix)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model = build_model(
        model_type=checkpoint["model_type"],
        input_dim=input_dim,
        seq_len=checkpoint["seq_len"],
        params=checkpoint["model_params"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)

    scaler = StandardScaler()
    scaler.mean_ = checkpoint["scaler_mean_"]
    scaler.scale_ = checkpoint["scaler_scale_"]

    return model, scaler, checkpoint


def build_sequence_bundle(
    file_name: str,
    seq_len: int = DEFAULT_SEQ_LEN,
    test_size: float = DEFAULT_TEST_SIZE,
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
    valid_size: int = DEFAULT_VALID_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
) -> SequenceBundle:
    df = load_dataset(file_name)
    target_col, task = get_target_and_task(file_name)
    feature_cols = get_feature_columns(df, target_col)

    X_raw = df[feature_cols].values.astype(np.float32)
    y_raw = df[target_col].values.astype(np.float32)
    dates = df["Date"].values

    X_seq, y_seq, seq_dates = create_sequences(
        X=X_raw,
        y=y_raw,
        dates=dates,
        seq_len=seq_len,
    )

    seq_df = pd.DataFrame({
        "Date": pd.to_datetime(seq_dates),
        "seq_idx": np.arange(len(seq_dates)),
    })

    dev_seq_df, test_seq_df = train_dev_test_split(seq_df, test_size=test_size)
    dev_idx = dev_seq_df["seq_idx"].to_numpy()
    test_idx = test_seq_df["seq_idx"].to_numpy()

    X_dev_raw = X_seq[dev_idx]
    y_dev = y_seq[dev_idx]
    X_test_raw = X_seq[test_idx]
    y_test = y_seq[test_idx]

    config = DATASET_CONFIG[file_name]
    purge = config["purge"]
    embargo = config["embargo"]

    folds = list(
        expanding_window_splits(
            n_samples=len(X_dev_raw),
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
            f"Check dataset size, seq_len, min_train_size, valid_size, step_size."
        )

    return SequenceBundle(
        file_name=file_name,
        target_col=target_col,
        task=task,
        feature_cols=feature_cols,
        seq_len=seq_len,
        seq_df=seq_df,
        X_dev_raw=X_dev_raw,
        y_dev=y_dev,
        X_test_raw=X_test_raw,
        y_test=y_test,
        folds=folds,
        n_features=X_dev_raw.shape[2],
    )


def fit_one_deep_model(
    model_type: str,
    model_params: Dict,
    input_dim: int,
    seq_len: int,
    task: str,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_valid_raw: np.ndarray,
    y_valid: np.ndarray,
    training_config: Dict,
) -> Tuple[torch.nn.Module, StandardScaler, Dict[str, float], float]:
    X_train, X_valid, _, scaler = scale_datasets(
        X_train_raw=X_train_raw,
        X_valid_raw=X_valid_raw,
        X_test_raw=None,
    )

    train_loader = make_dataloader(
        X=X_train,
        y=y_train,
        batch_size=training_config["batch_size"],
        shuffle=True,
    )
    valid_loader = make_dataloader(
        X=X_valid,
        y=y_valid,
        batch_size=training_config["batch_size"],
        shuffle=False,
    )

    model = build_model(
        model_type=model_type,
        input_dim=input_dim,
        seq_len=seq_len,
        params=model_params,
    )

    model, valid_metrics, valid_loss = train_one_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        task=task,
        y_train=y_train,
        epochs=training_config["epochs"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        patience=training_config["patience"],
        min_delta=training_config["min_delta"],
        grad_clip_norm=training_config["grad_clip_norm"],
    )

    return model, scaler, valid_metrics, valid_loss


def evaluate_model_on_test_with_scaler(
    model: torch.nn.Module,
    scaler: StandardScaler,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    task: str,
    batch_size: int,
) -> Dict[str, float]:
    X_test = apply_feature_scaler(scaler, X_test_raw)

    test_loader = make_dataloader(
        X=X_test,
        y=y_test,
        batch_size=batch_size,
        shuffle=False,
    )
    _, test_metrics = evaluate_model(model, test_loader, task)
    return test_metrics


def run_deep_cv_for_dataset(
    file_name: str,
    seq_len: int = DEFAULT_SEQ_LEN,
    test_size: float = DEFAULT_TEST_SIZE,
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
    valid_size: int = DEFAULT_VALID_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
    skip_existing: bool = DEFAULT_SKIP_EXISTING,
    force_retrain: bool = False,
    save_dir: str = MODEL_DIR,
    result_dir: str = RESULTS_DIR,
) -> pd.DataFrame:
    ensure_dirs()

    bundle = build_sequence_bundle(
        file_name=file_name,
        seq_len=seq_len,
        test_size=test_size,
        min_train_size=min_train_size,
        valid_size=valid_size,
        step_size=step_size,
    )

    results = []

    for model_name, (model_type, model_params) in get_deep_models().items():
        checkpoint_path = model_checkpoint_path(bundle.file_name, model_name, save_dir=save_dir)

        if skip_existing and (not force_retrain) and os.path.exists(checkpoint_path):
            print(f"[SKIP TRAIN] {bundle.file_name} / {model_name} already trained")

            model, scaler, checkpoint = load_checkpoint(
                dataset_name=bundle.file_name,
                model_name=model_name,
                input_dim=bundle.n_features,
                seq_len=bundle.seq_len,
                save_dir=save_dir,
            )

            test_metrics = evaluate_model_on_test_with_scaler(
                model=model,
                scaler=scaler,
                X_test_raw=bundle.X_test_raw,
                y_test=bundle.y_test,
                task=bundle.task,
                batch_size=TRAINING_CONFIG["batch_size"],
            )

            row = {
                "dataset": bundle.file_name,
                "task": bundle.task,
                "model": model_name,
                "model_family": "deep_learning",
                "trained_this_run": False,
                "model_path": checkpoint_path,
                "n_dev": len(bundle.X_dev_raw),
                "n_test": len(bundle.X_test_raw),
                "n_folds": len(bundle.folds),
                "seq_len": bundle.seq_len,
                "input_dim": bundle.n_features,
            }
            for k, v in test_metrics.items():
                row[f"test_{k}"] = v

            results.append(row)
            continue

        fold_records = []

        for i, fold in enumerate(bundle.folds, start=1):
            X_train_raw = bundle.X_dev_raw[fold.train_idx]
            y_train = bundle.y_dev[fold.train_idx]
            X_valid_raw = bundle.X_dev_raw[fold.valid_idx]
            y_valid = bundle.y_dev[fold.valid_idx]

            _, _, valid_metrics, valid_loss = fit_one_deep_model(
                model_type=model_type,
                model_params=model_params,
                input_dim=bundle.n_features,
                seq_len=bundle.seq_len,
                task=bundle.task,
                X_train_raw=X_train_raw,
                y_train=y_train,
                X_valid_raw=X_valid_raw,
                y_valid=y_valid,
                training_config=TRAINING_CONFIG,
            )

            rec = {
                "fold": i,
                "fold_start": int(fold.valid_idx[0]),
                "fold_end": int(fold.valid_idx[-1]),
                "valid_loss": float(valid_loss),
            }
            rec.update(valid_metrics)
            fold_records.append(rec)

        cv_summary = summarize_cv(fold_records)

        X_final_train_raw, y_final_train, X_final_valid_raw, y_final_valid = split_dev_for_final_training(
            X_dev_raw=bundle.X_dev_raw,
            y_dev=bundle.y_dev,
            final_valid_size=FINAL_VALID_SIZE,
        )

        final_model, final_scaler, _, _ = fit_one_deep_model(
            model_type=model_type,
            model_params=model_params,
            input_dim=bundle.n_features,
            seq_len=bundle.seq_len,
            task=bundle.task,
            X_train_raw=X_final_train_raw,
            y_train=y_final_train,
            X_valid_raw=X_final_valid_raw,
            y_valid=y_final_valid,
            training_config=TRAINING_CONFIG,
        )

        checkpoint_path = save_checkpoint(
            model=final_model,
            dataset_name=bundle.file_name,
            model_name=model_name,
            task=bundle.task,
            seq_len=bundle.seq_len,
            feature_cols=bundle.feature_cols,
            scaler=final_scaler,
            model_type=model_type,
            model_params=model_params,
            train_config=TRAINING_CONFIG,
            save_dir=save_dir,
            extra_meta={
                "training_mode": "standard",
                "test_size": test_size,
                "min_train_size": min_train_size,
                "valid_size": valid_size,
                "step_size": step_size,
            },
        )

        test_metrics = evaluate_model_on_test_with_scaler(
            model=final_model,
            scaler=final_scaler,
            X_test_raw=bundle.X_test_raw,
            y_test=bundle.y_test,
            task=bundle.task,
            batch_size=TRAINING_CONFIG["batch_size"],
        )

        row = {
            "dataset": bundle.file_name,
            "task": bundle.task,
            "model": model_name,
            "model_family": "deep_learning",
            "trained_this_run": True,
            "model_path": checkpoint_path,
            "n_dev": len(bundle.X_dev_raw),
            "n_test": len(bundle.X_test_raw),
            "n_folds": len(bundle.folds),
            "seq_len": bundle.seq_len,
            "input_dim": bundle.n_features,
            **cv_summary,
        }
        for k, v in test_metrics.items():
            row[f"test_{k}"] = v

        results.append(row)

    out_df = pd.DataFrame(results)

    os.makedirs(result_dir, exist_ok=True)
    out_path = os.path.join(
        result_dir,
        f"deep_model_comparison_{file_name.replace('.csv', '')}.csv",
    )
    out_df.to_csv(out_path, index=False)
    print(f"Saved deep results to {out_path}")

    return out_df


def run_all_deep_datasets(
    seq_len: int = DEFAULT_SEQ_LEN,
    skip_existing: bool = DEFAULT_SKIP_EXISTING,
    force_retrain: bool = False,
    save_dir: str = MODEL_DIR,
    result_filename: str = "deep_model_comparison_expanding.csv",
) -> pd.DataFrame:
    ensure_dirs()

    frames = []
    for file_name in DATASET_CONFIG:
        print(f"Running deep dataset: {file_name}")
        df = run_deep_cv_for_dataset(
            file_name=file_name,
            seq_len=seq_len,
            skip_existing=skip_existing,
            force_retrain=force_retrain,
            save_dir=save_dir,
            result_dir=RESULTS_DIR,
        )
        frames.append(df)

    final_df = pd.concat(frames, axis=0, ignore_index=True)
    out_path = os.path.join(RESULTS_DIR, result_filename)
    final_df.to_csv(out_path, index=False)
    print(f"Saved combined deep results to {out_path}")
    return final_df


# -------------------------------------------------------------------
# Bootstrap retraining
# -------------------------------------------------------------------

def fit_bootstrap_deep_model_and_score(
    bundle: SequenceBundle,
    model_name: str,
    model_type: str,
    model_params: Dict,
    bootstrap_sample: BootstrapSample,
    bootstrap_id: int,
    save_models: bool,
    save_dir: str,
) -> Dict[str, object]:
    boot_idx = bootstrap_sample.sample_idx

    X_boot_raw = bundle.X_dev_raw[boot_idx]
    y_boot = bundle.y_dev[boot_idx]

    X_final_train_raw, y_final_train, X_final_valid_raw, y_final_valid = split_dev_for_final_training(
        X_dev_raw=X_boot_raw,
        y_dev=y_boot,
        final_valid_size=FINAL_VALID_SIZE,
    )

    model, scaler, valid_metrics, valid_loss = fit_one_deep_model(
        model_type=model_type,
        model_params=model_params,
        input_dim=bundle.n_features,
        seq_len=bundle.seq_len,
        task=bundle.task,
        X_train_raw=X_final_train_raw,
        y_train=y_final_train,
        X_valid_raw=X_final_valid_raw,
        y_valid=y_final_valid,
        training_config=TRAINING_CONFIG,
    )

    if save_models:
        checkpoint_path = save_checkpoint(
            model=model,
            dataset_name=bundle.file_name,
            model_name=model_name,
            task=bundle.task,
            seq_len=bundle.seq_len,
            feature_cols=bundle.feature_cols,
            scaler=scaler,
            model_type=model_type,
            model_params=model_params,
            train_config=TRAINING_CONFIG,
            save_dir=save_dir,
            suffix=f"bootstrap_{bootstrap_id:04d}",
            extra_meta={
                "training_mode": "bootstrap_retrain",
                "bootstrap_id": bootstrap_id,
                "bootstrap_method": bootstrap_sample.method,
                "block_length": bootstrap_sample.block_length,
                "bootstrap_seed": bootstrap_sample.seed,
            },
        )
    else:
        checkpoint_path = ""

    test_metrics = evaluate_model_on_test_with_scaler(
        model=model,
        scaler=scaler,
        X_test_raw=bundle.X_test_raw,
        y_test=bundle.y_test,
        task=bundle.task,
        batch_size=TRAINING_CONFIG["batch_size"],
    )

    row = {
        "dataset": bundle.file_name,
        "task": bundle.task,
        "model": model_name,
        "model_family": "deep_learning",
        "training_mode": "bootstrap_retrain",
        "bootstrap_id": int(bootstrap_id),
        "bootstrap_method": bootstrap_sample.method,
        "block_length": int(bootstrap_sample.block_length),
        "bootstrap_seed": int(bootstrap_sample.seed) if bootstrap_sample.seed is not None else np.nan,
        "bootstrap_unique_fraction": float(bootstrap_sample.unique_fraction),
        "seq_len": bundle.seq_len,
        "input_dim": bundle.n_features,
        "trained_this_run": True,
        "model_path": checkpoint_path,
        "n_dev_original": len(bundle.X_dev_raw),
        "n_dev_bootstrap": len(X_boot_raw),
        "n_test": len(bundle.X_test_raw),
        "valid_loss": float(valid_loss),
    }
    row.update({f"valid_{k}": v for k, v in valid_metrics.items()})
    row.update({f"test_{k}": v for k, v in test_metrics.items()})
    return row


def run_deep_bootstrap_for_dataset(
    file_name: str,
    seq_len: int = DEFAULT_SEQ_LEN,
    test_size: float = DEFAULT_TEST_SIZE,
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
    valid_size: int = DEFAULT_VALID_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
    n_bootstrap: int = 30,
    bootstrap_method: str = "stationary",
    block_length: Optional[int] = None,
    base_seed: int = 42,
    save_models: bool = False,
    save_dir: str = ROBUSTNESS_DIR,
) -> pd.DataFrame:
    ensure_dirs()

    bundle = build_sequence_bundle(
        file_name=file_name,
        seq_len=seq_len,
        test_size=test_size,
        min_train_size=min_train_size,
        valid_size=valid_size,
        step_size=step_size,
    )

    if block_length is None:
        block_length = suggest_block_length(len(bundle.X_dev_raw), rule="sqrt", min_block_length=5)

    rows = []

    for model_name, (model_type, model_params) in get_deep_models().items():
        print(
            f"[DEEP BOOTSTRAP] dataset={bundle.file_name} model={model_name} "
            f"method={bootstrap_method} block_length={block_length} n_bootstrap={n_bootstrap}"
        )

        for b, sample in enumerate(
            iter_bootstrap_samples(
                n_samples=len(bundle.X_dev_raw),
                n_bootstrap=n_bootstrap,
                method=bootstrap_method,
                block_length=block_length,
                base_seed=base_seed,
            ),
            start=1,
        ):
            row = fit_bootstrap_deep_model_and_score(
                bundle=bundle,
                model_name=model_name,
                model_type=model_type,
                model_params=model_params,
                bootstrap_sample=sample,
                bootstrap_id=b,
                save_models=save_models,
                save_dir=save_dir,
            )
            rows.append(row)

    out_df = pd.DataFrame(rows)

    out_path = os.path.join(
        ROBUSTNESS_RESULTS_DIR,
        f"deep_bootstrap_{file_name.replace('.csv', '')}_{bootstrap_method}.csv",
    )
    out_df.to_csv(out_path, index=False)
    print(f"Saved deep bootstrap details to {out_path}")

    summary_df = summarize_deep_bootstrap_results(out_df)
    summary_path = os.path.join(
        ROBUSTNESS_RESULTS_DIR,
        f"deep_bootstrap_summary_{file_name.replace('.csv', '')}_{bootstrap_method}.csv",
    )
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved deep bootstrap summary to {summary_path}")

    return out_df


def summarize_deep_bootstrap_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    group_cols = ["dataset", "task", "model", "model_family", "training_mode", "bootstrap_method", "block_length"]

    numeric_cols = [
        col for col in df.columns
        if (col.startswith("test_") or col.startswith("valid_")) and pd.api.types.is_numeric_dtype(df[col])
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


def run_deep_bootstrap_all_datasets(
    seq_len: int = DEFAULT_SEQ_LEN,
    n_bootstrap: int = 30,
    bootstrap_method: str = "stationary",
    block_length: Optional[int] = None,
    base_seed: int = 42,
    save_models: bool = False,
    save_dir: str = ROBUSTNESS_DIR,
) -> pd.DataFrame:
    ensure_dirs()

    frames = []
    for file_name in DATASET_CONFIG:
        df = run_deep_bootstrap_for_dataset(
            file_name=file_name,
            seq_len=seq_len,
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
        f"deep_bootstrap_all_{bootstrap_method}.csv",
    )
    final_df.to_csv(out_path, index=False)
    print(f"Saved combined deep bootstrap details to {out_path}")

    summary_df = summarize_deep_bootstrap_results(final_df)
    summary_path = os.path.join(
        ROBUSTNESS_RESULTS_DIR,
        f"deep_bootstrap_all_summary_{bootstrap_method}.csv",
    )
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved combined deep bootstrap summary to {summary_path}")

    return final_df


# -------------------------------------------------------------------
# Walk-forward detail
# -------------------------------------------------------------------

def run_deep_walk_forward_detail_for_dataset(
    file_name: str,
    seq_len: int = DEFAULT_SEQ_LEN,
    test_size: float = DEFAULT_TEST_SIZE,
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
    valid_size: int = DEFAULT_VALID_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
) -> pd.DataFrame:
    ensure_dirs()

    bundle = build_sequence_bundle(
        file_name=file_name,
        seq_len=seq_len,
        test_size=test_size,
        min_train_size=min_train_size,
        valid_size=valid_size,
        step_size=step_size,
    )

    rows = []

    for model_name, (model_type, model_params) in get_deep_models().items():
        for i, fold in enumerate(bundle.folds, start=1):
            X_train_raw = bundle.X_dev_raw[fold.train_idx]
            y_train = bundle.y_dev[fold.train_idx]
            X_valid_raw = bundle.X_dev_raw[fold.valid_idx]
            y_valid = bundle.y_dev[fold.valid_idx]

            _, _, valid_metrics, valid_loss = fit_one_deep_model(
                model_type=model_type,
                model_params=model_params,
                input_dim=bundle.n_features,
                seq_len=bundle.seq_len,
                task=bundle.task,
                X_train_raw=X_train_raw,
                y_train=y_train,
                X_valid_raw=X_valid_raw,
                y_valid=y_valid,
                training_config=TRAINING_CONFIG,
            )

            row = {
                "dataset": bundle.file_name,
                "task": bundle.task,
                "model": model_name,
                "model_family": "deep_learning",
                "fold": i,
                "train_start_idx": int(fold.train_idx[0]),
                "train_end_idx": int(fold.train_idx[-1]),
                "valid_start_idx": int(fold.valid_idx[0]),
                "valid_end_idx": int(fold.valid_idx[-1]),
                "train_size_effective": int(len(fold.train_idx)),
                "valid_size": int(len(fold.valid_idx)),
                "valid_start_date": str(bundle.seq_df.iloc[fold.valid_idx[0]]["Date"].date()),
                "valid_end_date": str(bundle.seq_df.iloc[fold.valid_idx[-1]]["Date"].date()),
                "seq_len": bundle.seq_len,
                "input_dim": bundle.n_features,
                "valid_loss": float(valid_loss),
            }
            row.update(valid_metrics)
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(
        ROBUSTNESS_RESULTS_DIR,
        f"deep_walk_forward_detail_{file_name.replace('.csv', '')}.csv",
    )
    out_df.to_csv(out_path, index=False)
    print(f"Saved deep walk-forward detail to {out_path}")
    return out_df


if __name__ == "__main__":
    df = run_all_deep_datasets(
        seq_len=DEFAULT_SEQ_LEN,
        skip_existing=True,
        force_retrain=False,
        save_dir=MODEL_DIR,
        result_filename="deep_model_comparison_expanding.csv",
    )
    print(df)