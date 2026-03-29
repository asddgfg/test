# cross_task_consistency.py
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


RESULTS_DIR = "results"
ROBUSTNESS_RESULTS_DIR = os.path.join(RESULTS_DIR, "robustness")


# ------------------------------------------------------------
# File helpers
# ------------------------------------------------------------

def ensure_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ROBUSTNESS_RESULTS_DIR, exist_ok=True)


def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_standard_results() -> pd.DataFrame:
    """
    Load standard model comparison outputs from:
    - classical
    - deep
    - TFT

    The function looks for the most likely consolidated result files first.
    """
    candidates = [
        os.path.join(RESULTS_DIR, "model_comparison_expanding.csv"),
        os.path.join(RESULTS_DIR, "deep_model_comparison_expanding.csv"),
        os.path.join(RESULTS_DIR, "tft_model_comparison_expanding.csv"),
    ]

    frames = []
    for p in candidates:
        df = _safe_read_csv(p)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=0, ignore_index=True)

    # Standardize missing columns if needed
    if "model_family" not in out.columns:
        out["model_family"] = "unknown"

    return out


def load_bootstrap_summary() -> pd.DataFrame:
    path = os.path.join(ROBUSTNESS_RESULTS_DIR, "bootstrap_summary_all_models.csv")
    return _safe_read_csv(path)


def load_regime_analysis() -> pd.DataFrame:
    path = os.path.join(ROBUSTNESS_RESULTS_DIR, "regime_analysis_all_models.csv")
    return _safe_read_csv(path)


def load_walk_forward_detail() -> pd.DataFrame:
    path = os.path.join(ROBUSTNESS_RESULTS_DIR, "walk_forward_detail_all_models.csv")
    return _safe_read_csv(path)


# ------------------------------------------------------------
# Metric selection
# ------------------------------------------------------------

def task_type_from_row(row: pd.Series) -> str:
    return str(row.get("task", "")).strip().lower()


def choose_primary_metric_for_task(task: str, available_cols: List[str]) -> Optional[str]:
    """
    Pick one primary metric per task for cross-task comparison.

    Regression:
        prefer test_oos_r2, else test_rmse, else test_mae, else directional accuracy

    Classification:
        prefer test_roc_auc, else test_f1, else test_accuracy

    Returns the chosen column name or None.
    """
    task = task.lower()

    if task == "regression":
        for col in ["test_oos_r2", "test_rmse", "test_mae", "test_directional_accuracy"]:
            if col in available_cols:
                return col

    if task == "classification":
        for col in ["test_roc_auc", "test_f1", "test_accuracy"]:
            if col in available_cols:
                return col

    return None


def metric_higher_is_better(metric_name: str) -> bool:
    metric_name = str(metric_name).lower()
    loss_like = ["rmse", "mae", "log_loss", "brier"]
    return not any(x in metric_name for x in loss_like)


def signed_metric_value(metric_name: str, value: float) -> float:
    """
    Convert all metrics so 'higher is better' for ranking consistency.
    For loss-like metrics, multiply by -1.
    """
    if pd.isna(value):
        return np.nan
    return float(value) if metric_higher_is_better(metric_name) else float(-value)


# ------------------------------------------------------------
# Standard cross-task consistency
# ------------------------------------------------------------

def build_primary_task_performance_table(standard_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per (dataset, task, model) with:
    - chosen primary metric
    - signed metric value
    """
    if standard_df.empty:
        return pd.DataFrame()

    rows = []
    available_cols = list(standard_df.columns)

    for _, row in standard_df.iterrows():
        task = str(row.get("task", "")).lower()
        metric_col = choose_primary_metric_for_task(task, available_cols)
        if metric_col is None:
            continue

        metric_val = row.get(metric_col, np.nan)
        signed_val = signed_metric_value(metric_col, metric_val)

        rows.append({
            "dataset": row.get("dataset"),
            "task": row.get("task"),
            "model": row.get("model"),
            "model_family": row.get("model_family", "unknown"),
            "primary_metric": metric_col,
            "primary_metric_value": metric_val,
            "primary_metric_signed": signed_val,
        })

    return pd.DataFrame(rows)


def summarize_cross_task_consistency(primary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize how stable each model is across tasks using:
    - average signed score
    - std of signed score
    - min / max signed score
    - number of tasks covered
    """
    if primary_df.empty:
        return pd.DataFrame()

    rows = []

    for (model, model_family), grp in primary_df.groupby(["model", "model_family"], dropna=False):
        vals = grp["primary_metric_signed"].dropna().astype(float).values

        row = {
            "model": model,
            "model_family": model_family,
            "n_tasks_covered": int(grp["task"].nunique()),
            "n_datasets_covered": int(grp["dataset"].nunique()),
            "cross_task_score_mean": float(np.mean(vals)) if len(vals) > 0 else np.nan,
            "cross_task_score_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "cross_task_score_min": float(np.min(vals)) if len(vals) > 0 else np.nan,
            "cross_task_score_max": float(np.max(vals)) if len(vals) > 0 else np.nan,
        }

        # Add per-task details as columns if present
        for task_name in sorted(grp["task"].dropna().unique()):
            sub = grp[grp["task"] == task_name]
            if len(sub) > 0:
                row[f"{task_name}_metric"] = sub.iloc[0]["primary_metric"]
                row[f"{task_name}_score_signed"] = sub.iloc[0]["primary_metric_signed"]
                row[f"{task_name}_score_raw"] = sub.iloc[0]["primary_metric_value"]

        rows.append(row)

    out = pd.DataFrame(rows)

    # Lower std = more stable, higher mean = better
    # A simple combined score:
    out["consistency_composite_score"] = (
        out["cross_task_score_mean"] - 0.5 * out["cross_task_score_std"]
    )

    out = out.sort_values(
        ["consistency_composite_score", "cross_task_score_mean"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return out


# ------------------------------------------------------------
# Bootstrap-enhanced consistency
# ------------------------------------------------------------

def pick_bootstrap_primary_metric_row(row: pd.Series) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """
    From a bootstrap summary row, choose a task-appropriate mean metric and std metric.
    """
    task = str(row.get("task", "")).lower()
    available_cols = list(row.index)

    if task == "regression":
        candidates = [
            ("test_oos_r2_mean", "test_oos_r2_std"),
            ("test_rmse_mean", "test_rmse_std"),
            ("test_mae_mean", "test_mae_std"),
            ("test_directional_accuracy_mean", "test_directional_accuracy_std"),
        ]
    elif task == "classification":
        candidates = [
            ("test_roc_auc_mean", "test_roc_auc_std"),
            ("test_f1_mean", "test_f1_std"),
            ("test_accuracy_mean", "test_accuracy_std"),
        ]
    else:
        candidates = []

    for mean_col, std_col in candidates:
        if mean_col in available_cols:
            mean_val = row.get(mean_col, np.nan)
            std_val = row.get(std_col, np.nan)
            return mean_col, mean_val, std_val

    return None, None, None


def summarize_bootstrap_cross_task_consistency(bootstrap_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each model across tasks:
    - mean bootstrap task performance
    - std across tasks
    - average within-task bootstrap uncertainty
    """
    if bootstrap_df.empty:
        return pd.DataFrame()

    rows = []
    task_rows = []

    for _, row in bootstrap_df.iterrows():
        metric_name, mean_val, std_val = pick_bootstrap_primary_metric_row(row)
        if metric_name is None:
            continue

        signed_mean = signed_metric_value(metric_name, mean_val)

        task_rows.append({
            "dataset": row.get("dataset"),
            "task": row.get("task"),
            "model": row.get("model"),
            "model_family": row.get("model_family"),
            "primary_bootstrap_metric": metric_name,
            "primary_bootstrap_mean": mean_val,
            "primary_bootstrap_std": std_val,
            "primary_bootstrap_signed_mean": signed_mean,
        })

    task_df = pd.DataFrame(task_rows)
    if task_df.empty:
        return pd.DataFrame()

    for (model, model_family), grp in task_df.groupby(["model", "model_family"], dropna=False):
        signed_vals = grp["primary_bootstrap_signed_mean"].dropna().astype(float).values
        uncert_vals = grp["primary_bootstrap_std"].dropna().astype(float).values

        row = {
            "model": model,
            "model_family": model_family,
            "n_tasks_covered": int(grp["task"].nunique()),
            "bootstrap_cross_task_mean": float(np.mean(signed_vals)) if len(signed_vals) > 0 else np.nan,
            "bootstrap_cross_task_std": float(np.std(signed_vals, ddof=1)) if len(signed_vals) > 1 else 0.0,
            "bootstrap_avg_within_task_uncertainty": float(np.mean(uncert_vals)) if len(uncert_vals) > 0 else np.nan,
        }

        row["bootstrap_consistency_composite_score"] = (
            row["bootstrap_cross_task_mean"]
            - 0.5 * row["bootstrap_cross_task_std"]
            - 0.25 * row["bootstrap_avg_within_task_uncertainty"]
            if pd.notna(row["bootstrap_cross_task_mean"])
            else np.nan
        )

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["bootstrap_consistency_composite_score", "bootstrap_cross_task_mean"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return out


# ------------------------------------------------------------
# Regime robustness consistency
# ------------------------------------------------------------

def summarize_regime_robustness(regime_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize how sensitive each model is to regime changes.

    For each model:
    - average absolute delta between regime metrics and overall metrics
    - lower = more regime-robust
    """
    if regime_df.empty:
        return pd.DataFrame()

    delta_cols = [c for c in regime_df.columns if c.endswith("_delta_vs_overall")]
    if not delta_cols:
        return pd.DataFrame()

    rows = []

    for (model, model_family), grp in regime_df.groupby(["model", "model_family"], dropna=False):
        abs_deltas = []

        for col in delta_cols:
            vals = pd.to_numeric(grp[col], errors="coerce").dropna().abs().values
            if len(vals) > 0:
                abs_deltas.extend(vals.tolist())

        row = {
            "model": model,
            "model_family": model_family,
            "n_regime_rows": int(len(grp)),
            "avg_abs_regime_delta": float(np.mean(abs_deltas)) if len(abs_deltas) > 0 else np.nan,
            "max_abs_regime_delta": float(np.max(abs_deltas)) if len(abs_deltas) > 0 else np.nan,
        }

        # Smaller regime delta is better
        row["regime_robustness_score"] = (
            -row["avg_abs_regime_delta"] if pd.notna(row["avg_abs_regime_delta"]) else np.nan
        )

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["regime_robustness_score", "avg_abs_regime_delta"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return out


# ------------------------------------------------------------
# Final merged leaderboard
# ------------------------------------------------------------

def build_final_consistency_leaderboard(
    standard_summary: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge:
    - standard cross-task consistency
    - bootstrap-enhanced consistency
    - regime robustness

    and produce one final leaderboard.
    """
    base = standard_summary.copy()

    if base.empty and bootstrap_summary.empty and regime_summary.empty:
        return pd.DataFrame()

    if base.empty:
        # Make a minimal base from whichever summary exists
        if not bootstrap_summary.empty:
            base = bootstrap_summary[["model", "model_family"]].drop_duplicates().copy()
        elif not regime_summary.empty:
            base = regime_summary[["model", "model_family"]].drop_duplicates().copy()

    if not bootstrap_summary.empty:
        keep_cols = [
            "model",
            "model_family",
            "bootstrap_cross_task_mean",
            "bootstrap_cross_task_std",
            "bootstrap_avg_within_task_uncertainty",
            "bootstrap_consistency_composite_score",
        ]
        base = base.merge(
            bootstrap_summary[keep_cols],
            on=["model", "model_family"],
            how="left",
        )

    if not regime_summary.empty:
        keep_cols = [
            "model",
            "model_family",
            "avg_abs_regime_delta",
            "max_abs_regime_delta",
            "regime_robustness_score",
        ]
        base = base.merge(
            regime_summary[keep_cols],
            on=["model", "model_family"],
            how="left",
        )

    # Final composite:
    # standard consistency + bootstrap consistency + regime robustness
    # all already oriented so higher is better
    comp_cols = [
        "consistency_composite_score",
        "bootstrap_consistency_composite_score",
        "regime_robustness_score",
    ]

    def row_mean_ignore_nan(r):
        vals = [r[c] for c in comp_cols if c in r.index and pd.notna(r[c])]
        return float(np.mean(vals)) if len(vals) > 0 else np.nan

    base["final_consistency_score"] = base.apply(row_mean_ignore_nan, axis=1)

    base = base.sort_values(
        ["final_consistency_score", "consistency_composite_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return base


# ------------------------------------------------------------
# Main runner
# ------------------------------------------------------------

def run_cross_task_consistency_analysis() -> Dict[str, pd.DataFrame]:
    ensure_dirs()

    standard_df = load_standard_results()
    bootstrap_df = load_bootstrap_summary()
    regime_df = load_regime_analysis()
    walk_df = load_walk_forward_detail()  # currently loaded for completeness, not used directly

    primary_df = build_primary_task_performance_table(standard_df)
    standard_summary = summarize_cross_task_consistency(primary_df)
    bootstrap_summary = summarize_bootstrap_cross_task_consistency(bootstrap_df)
    regime_summary = summarize_regime_robustness(regime_df)

    final_leaderboard = build_final_consistency_leaderboard(
        standard_summary=standard_summary,
        bootstrap_summary=bootstrap_summary,
        regime_summary=regime_summary,
    )

    outputs = {
        "primary_task_table": primary_df,
        "standard_summary": standard_summary,
        "bootstrap_summary": bootstrap_summary,
        "regime_summary": regime_summary,
        "final_leaderboard": final_leaderboard,
    }

    for name, df in outputs.items():
        if df is not None and not df.empty:
            path = os.path.join(ROBUSTNESS_RESULTS_DIR, f"{name}.csv")
            df.to_csv(path, index=False)
            print(f"Saved {name} to {path}")

    return outputs


if __name__ == "__main__":
    results = run_cross_task_consistency_analysis()
    for k, v in results.items():
        print(f"{k}: shape={v.shape}")