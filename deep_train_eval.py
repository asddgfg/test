import copy
import json
import os
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

from deep_learning_models import get_deep_models, build_model
from splitter import expanding_window_splits, train_dev_test_split
from train_eval import DATASET_CONFIG


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
MODEL_DIR = "models_saved_deep"

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
        if col != "fold":
            out[f"{col}_cv_mean"] = float(df[col].mean())
            out[f"{col}_cv_std"] = float(df[col].std())

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


def model_checkpoint_path(dataset_name: str, model_name: str) -> str:
    return os.path.join(MODEL_DIR, f"{dataset_name.replace('.csv', '')}__{model_name}.pt")


def model_meta_path(dataset_name: str, model_name: str) -> str:
    return os.path.join(MODEL_DIR, f"{dataset_name.replace('.csv', '')}__{model_name}.meta.json")


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
) -> str:
    path = model_checkpoint_path(dataset_name, model_name)

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
    with open(model_meta_path(dataset_name, model_name), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return path


def run_deep_cv_for_dataset(
    file_name: str,
    seq_len: int = DEFAULT_SEQ_LEN,
    test_size: float = DEFAULT_TEST_SIZE,
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
    valid_size: int = DEFAULT_VALID_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
    skip_existing: bool = DEFAULT_SKIP_EXISTING,
) -> pd.DataFrame:
    ensure_dirs()

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

    input_dim = X_dev_raw.shape[2]
    results = []

    for model_name, (model_type, model_params) in get_deep_models().items():
        checkpoint_path = model_checkpoint_path(file_name, model_name)
        if skip_existing and os.path.exists(checkpoint_path):
  
            print(f"[SKIP TRAIN] {file_name} / {model_name} already trained")

            # ----------------------------
            # LOAD MODEL
            # ----------------------------
            checkpoint = torch.load(
                checkpoint_path,
                map_location=DEVICE,
                weights_only=False,
            )

            model = build_model(
                model_type=checkpoint["model_type"],
                input_dim=X_dev_raw.shape[2],
                seq_len=checkpoint["seq_len"],
                params=checkpoint["model_params"],
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(DEVICE)

            # ----------------------------
            # REBUILD SCALER
            # ----------------------------
            scaler = StandardScaler()
            scaler.mean_ = checkpoint["scaler_mean_"]
            scaler.scale_ = checkpoint["scaler_scale_"]

            # scale test
            X_test = apply_feature_scaler(scaler, X_test_raw)

            test_loader = make_dataloader(
                X=X_test,
                y=y_test,
                batch_size=TRAINING_CONFIG["batch_size"],
                shuffle=False,
            )

            # ----------------------------
            # EVALUATE
            # ----------------------------
            _, test_metrics = evaluate_model(model, test_loader, task)

            row = {
                "dataset": file_name,
                "task": task,
                "model": model_name,
                "model_family": "deep_learning",
                "trained_this_run": False,
                "model_path": checkpoint_path,
                "n_dev": len(X_dev_raw),
                "n_test": len(X_test_raw),
                "n_folds": len(folds),
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }

            results.append(row)
            continue

        print(f"[DEEP] {file_name} / {model_name} / task={task}")
        fold_records = []

        # ----------------------------
        # Walk-forward CV on dev
        # ----------------------------
        for i, fold in enumerate(folds, start=1):
            X_train_raw = X_dev_raw[fold.train_idx]
            y_train = y_dev[fold.train_idx]

            X_valid_raw = X_dev_raw[fold.valid_idx]
            y_valid = y_dev[fold.valid_idx]

            X_train, X_valid, _, _ = scale_datasets(
                X_train_raw=X_train_raw,
                X_valid_raw=X_valid_raw,
                X_test_raw=None,
            )

            train_loader = make_dataloader(
                X=X_train,
                y=y_train,
                batch_size=TRAINING_CONFIG["batch_size"],
                shuffle=True,
            )
            valid_loader = make_dataloader(
                X=X_valid,
                y=y_valid,
                batch_size=TRAINING_CONFIG["batch_size"],
                shuffle=False,
            )

            model = build_model(
                model_type=model_type,
                input_dim=input_dim,
                seq_len=seq_len,
                params=model_params,
            )

            _, valid_metrics, _ = train_one_model(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                task=task,
                y_train=y_train,
                epochs=TRAINING_CONFIG["epochs"],
                learning_rate=TRAINING_CONFIG["learning_rate"],
                weight_decay=TRAINING_CONFIG["weight_decay"],
                patience=TRAINING_CONFIG["patience"],
                min_delta=TRAINING_CONFIG["min_delta"],
                grad_clip_norm=TRAINING_CONFIG["grad_clip_norm"],
            )

            valid_metrics["fold"] = i
            fold_records.append(valid_metrics)

        cv_summary = summarize_cv(fold_records)

        # ----------------------------
        # Final train / final valid / untouched test
        # ----------------------------
        X_final_train_raw, y_final_train, X_final_valid_raw, y_final_valid = split_dev_for_final_training(
            X_dev_raw=X_dev_raw,
            y_dev=y_dev,
            final_valid_size=FINAL_VALID_SIZE,
        )

        X_final_train, X_final_valid, X_test, scaler = scale_datasets(
            X_train_raw=X_final_train_raw,
            X_valid_raw=X_final_valid_raw,
            X_test_raw=X_test_raw,
        )

        final_train_loader = make_dataloader(
            X=X_final_train,
            y=y_final_train,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=True,
        )
        final_valid_loader = make_dataloader(
            X=X_final_valid,
            y=y_final_valid,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=False,
        )
        test_loader = make_dataloader(
            X=X_test,
            y=y_test,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=False,
        )

        final_model = build_model(
            model_type=model_type,
            input_dim=input_dim,
            seq_len=seq_len,
            params=model_params,
        )

        final_model, _, _ = train_one_model(
            model=final_model,
            train_loader=final_train_loader,
            valid_loader=final_valid_loader,
            task=task,
            y_train=y_final_train,
            epochs=TRAINING_CONFIG["epochs"],
            learning_rate=TRAINING_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
            patience=TRAINING_CONFIG["patience"],
            min_delta=TRAINING_CONFIG["min_delta"],
            grad_clip_norm=TRAINING_CONFIG["grad_clip_norm"],
        )

        _, test_metrics = evaluate_model(final_model, test_loader, task)

        model_path = save_checkpoint(
            model=final_model,
            dataset_name=file_name,
            model_name=model_name,
            task=task,
            seq_len=seq_len,
            feature_cols=feature_cols,
            scaler=scaler,
            model_type=model_type,
            model_params=model_params,
            train_config={
                **TRAINING_CONFIG,
                "test_size": test_size,
                "min_train_size": min_train_size,
                "valid_size": valid_size,
                "step_size": step_size,
                "seq_len": seq_len,
                "purge": purge,
                "embargo": embargo,
            },
        )

        row = {
            "dataset": file_name,
            "task": task,
            "model": model_name,
            "model_family": "deep_learning",
            "trained_this_run": True,
            "model_path": model_path,
            "n_dev": len(X_dev_raw),
            "n_test": len(X_test_raw),
            "n_folds": len(folds),
            **cv_summary,
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        results.append(row)

    return pd.DataFrame(results)


def run_all_deep(
    seq_len: int = DEFAULT_SEQ_LEN,
    skip_existing: bool = DEFAULT_SKIP_EXISTING,
) -> pd.DataFrame:
    ensure_dirs()

    frames = []
    for file_name in DATASET_CONFIG.keys():
        result_df = run_deep_cv_for_dataset(
            file_name=file_name,
            seq_len=seq_len,
            skip_existing=skip_existing,
        )
        if not result_df.empty:
            frames.append(result_df)

    if not frames:
        print("No new deep-learning models were trained.")
        return pd.DataFrame()

    out = pd.concat(frames, axis=0, ignore_index=True)
    out_path = os.path.join(RESULTS_DIR, "deep_model_comparison.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved deep-learning results to {out_path}")
    return out


if __name__ == "__main__":
    df = run_all_deep()
    print(df)
