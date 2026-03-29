# main.py
from __future__ import annotations

import os
import shutil
import pandas as pd

from feature_engineering import run_feature_pipeline

# -----------------------------
# Standard pipeline
# -----------------------------
from train_eval import (
    run_all_datasets,
    DATASET_CONFIG,
    get_expected_model_names,
    model_exists,
)
from tft_train_eval import run_all_tft_datasets
from deep_train_eval import run_all_deep_datasets

# -----------------------------
# Robustness pipeline
# -----------------------------
from robustness_retrain import (
    run_classical_bootstrap,
    run_deep_bootstrap,
    run_tft_bootstrap,
    run_classical_walk_forward,
    run_deep_walk_forward,
    run_tft_walk_forward,
    run_classical_regime,
    run_deep_regime,
    run_tft_regime,
    aggregate_bootstrap_details,
    aggregate_bootstrap_summaries,
    aggregate_walk_forward_details,
    aggregate_regime_details,
)

from cross_task_consistency import run_cross_task_consistency_analysis


PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
ROBUSTNESS_RESULTS_DIR = os.path.join(RESULTS_DIR, "robustness")

MODEL_DIR = "models_saved"
MODEL_DIR_TFT = "models_saved_tft"
MODEL_DIR_DEEP = "models_saved_deep"
BEST_DIR = "models_best"

REQUIRED_DATASETS = list(DATASET_CONFIG.keys())

# -----------------------------
# Main pipeline controls
# -----------------------------
RUN_STANDARD_PIPELINE = True
RUN_BOOTSTRAP_ROBUSTNESS = True
RUN_WALK_FORWARD_DETAIL = True
RUN_REGIME_ANALYSIS = True
RUN_CROSS_TASK_CONSISTENCY = True

# Standard training behavior
FORCE_RETRAIN_STANDARD = False
SKIP_EXISTING_DEEP_MODELS = True

# Robustness settings
ROBUSTNESS_INCLUDE_CLASSICAL = True
ROBUSTNESS_INCLUDE_DEEP = True
ROBUSTNESS_INCLUDE_TFT = True

N_BOOTSTRAP = 5
BOOTSTRAP_METHOD = "stationary"   # "stationary", "moving", "circular"
BOOTSTRAP_BLOCK_LENGTH = None     # None -> auto choose in submodules
BOOTSTRAP_BASE_SEED = 42
SAVE_BOOTSTRAP_MODELS = False

DEEP_SEQ_LEN = 20


def ensure_dirs() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ROBUSTNESS_RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR_TFT, exist_ok=True)
    os.makedirs(MODEL_DIR_DEEP, exist_ok=True)


def processed_datasets_exist() -> bool:
    for file_name in REQUIRED_DATASETS:
        path = os.path.join(PROCESSED_DIR, file_name)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return False
    return True


def count_expected_classical_models() -> tuple[int, int]:
    expected = 0
    existing = 0

    for dataset_name, config in DATASET_CONFIG.items():
        task = config["task"]
        model_names = get_expected_model_names(task)

        for model_name in model_names:
            expected += 1
            if model_exists(dataset_name, model_name):
                existing += 1

    return existing, expected


def all_classical_models_exist() -> bool:
    existing, expected = count_expected_classical_models()
    return existing == expected


def print_pipeline_summary() -> None:
    print("===== PIPELINE CONFIGURATION =====")
    print(f"Datasets: {len(DATASET_CONFIG)}")
    print()

    for dataset_name, config in DATASET_CONFIG.items():
        task = config["task"]
        n_models = len(get_expected_model_names(task))
        print(
            f"  - {dataset_name}: "
            f"task={task}, "
            f"classical_models={n_models}, "
            f"purge={config['purge']}, "
            f"embargo={config['embargo']}"
        )

    existing, expected = count_expected_classical_models()
    print()
    print(f"Saved classical ML models found: {existing}/{expected}")
    print("TFT datasets: all datasets in TFT_DATASET_CONFIG")
    print("Deep learning datasets: all 4 datasets")
    print("----------------------------------")
    print(f"RUN_STANDARD_PIPELINE       = {RUN_STANDARD_PIPELINE}")
    print(f"RUN_BOOTSTRAP_ROBUSTNESS    = {RUN_BOOTSTRAP_ROBUSTNESS}")
    print(f"RUN_WALK_FORWARD_DETAIL     = {RUN_WALK_FORWARD_DETAIL}")
    print(f"RUN_REGIME_ANALYSIS         = {RUN_REGIME_ANALYSIS}")
    print(f"RUN_CROSS_TASK_CONSISTENCY  = {RUN_CROSS_TASK_CONSISTENCY}")
    print("----------------------------------")
    print(f"N_BOOTSTRAP                 = {N_BOOTSTRAP}")
    print(f"BOOTSTRAP_METHOD            = {BOOTSTRAP_METHOD}")
    print(f"BOOTSTRAP_BLOCK_LENGTH      = {BOOTSTRAP_BLOCK_LENGTH}")
    print(f"BOOTSTRAP_BASE_SEED         = {BOOTSTRAP_BASE_SEED}")
    print(f"SAVE_BOOTSTRAP_MODELS       = {SAVE_BOOTSTRAP_MODELS}")
    print(f"DEEP_SEQ_LEN                = {DEEP_SEQ_LEN}")
    print("==================================")


def remove_old_outputs() -> None:
    """
    Optional helper for a clean rerun.
    Not called automatically.
    """
    for folder in [RESULTS_DIR, MODEL_DIR, MODEL_DIR_TFT, MODEL_DIR_DEEP, BEST_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Removed folder: {folder}")


def run_data_preparation() -> None:
    print("===== STEP 1: DATA PREPARATION =====")
    if processed_datasets_exist():
        print("[1/8] Processed datasets already exist. Skipping data preparation.")
    else:
        print("[1/8] Missing processed datasets. Running data preparation...")
        output_paths = run_feature_pipeline()
        print("Saved processed datasets:")
        for name, path in output_paths.items():
            print(f"  {name}: {path}")


def run_standard_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("===== STEP 2-5: STANDARD MODEL PIPELINE =====")

    existing_before, expected_total = count_expected_classical_models()

    if all_classical_models_exist() and not FORCE_RETRAIN_STANDARD:
        print(f"[2/8] All expected classical ML models already exist ({existing_before}/{expected_total}).")
        print("      Skipping retraining where possible and running evaluation.")
    else:
        missing = expected_total - existing_before
        print(f"[2/8] Missing trained classical ML models: {missing}/{expected_total}")
        print("      Running training/evaluation for classical ML models...")

    print("[2B/8] Running classical ML evaluation / result aggregation...")
    ml_results = run_all_datasets(
        split_mode="expanding",
        force_retrain=FORCE_RETRAIN_STANDARD,
        save_dir=MODEL_DIR,
        result_filename="model_comparison_expanding.csv",
    )
    print(ml_results.head())

    print("[3/8] Running TFT evaluation / result aggregation...")
    tft_results = run_all_tft_datasets(
        force_retrain=FORCE_RETRAIN_STANDARD,
        save_dir=MODEL_DIR_TFT,
        result_filename="tft_model_comparison_expanding.csv",
        result_dir=RESULTS_DIR,
    )
    print(tft_results.head())

    print("[4/8] Running deep learning evaluation / result aggregation...")
    deep_results = run_all_deep_datasets(
        seq_len=DEEP_SEQ_LEN,
        skip_existing=SKIP_EXISTING_DEEP_MODELS,
        force_retrain=FORCE_RETRAIN_STANDARD,
        save_dir=MODEL_DIR_DEEP,
        result_filename="deep_model_comparison_expanding.csv",
    )
    print(deep_results.head())

    print("[5/8] Combining all standard model results...")

    if "task" not in deep_results.columns:
        task_map = {
            "dataset_return.csv": "regression",
            "dataset_direction.csv": "classification",
            "dataset_volatility.csv": "regression",
            "dataset_regime.csv": "classification",
        }
        deep_results = deep_results.copy()
        deep_results["task"] = deep_results["dataset"].map(task_map)

    all_results = pd.concat(
        [ml_results, tft_results, deep_results],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    all_results_path = os.path.join(RESULTS_DIR, "all_model_comparison_expanding.csv")
    all_results.to_csv(all_results_path, index=False)
    print(f"Saved combined standard results to {all_results_path}")

    return ml_results, tft_results, deep_results, all_results


def run_bootstrap_pipeline() -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    print("===== STEP 6A: BOOTSTRAP ROBUSTNESS =====")

    classical_bootstrap_df = None
    deep_bootstrap_df = None
    tft_bootstrap_df = None

    if ROBUSTNESS_INCLUDE_CLASSICAL:
        classical_bootstrap_df = run_classical_bootstrap(
            n_bootstrap=N_BOOTSTRAP,
            bootstrap_method=BOOTSTRAP_METHOD,
            block_length=BOOTSTRAP_BLOCK_LENGTH,
            base_seed=BOOTSTRAP_BASE_SEED,
            save_models=SAVE_BOOTSTRAP_MODELS,
        )

    if ROBUSTNESS_INCLUDE_DEEP:
        deep_bootstrap_df = run_deep_bootstrap(
            n_bootstrap=N_BOOTSTRAP,
            bootstrap_method=BOOTSTRAP_METHOD,
            block_length=BOOTSTRAP_BLOCK_LENGTH,
            base_seed=BOOTSTRAP_BASE_SEED,
            save_models=SAVE_BOOTSTRAP_MODELS,
            seq_len=DEEP_SEQ_LEN,
        )

    if ROBUSTNESS_INCLUDE_TFT:
        tft_bootstrap_df = run_tft_bootstrap(
            n_bootstrap=N_BOOTSTRAP,
            bootstrap_method=BOOTSTRAP_METHOD,
            block_length=BOOTSTRAP_BLOCK_LENGTH,
            base_seed=BOOTSTRAP_BASE_SEED,
            save_models=SAVE_BOOTSTRAP_MODELS,
        )

    aggregate_bootstrap_details(
        classical_df=classical_bootstrap_df,
        deep_df=deep_bootstrap_df,
        tft_df=tft_bootstrap_df,
    )
    aggregate_bootstrap_summaries(
        classical_df=classical_bootstrap_df,
        deep_df=deep_bootstrap_df,
        tft_df=tft_bootstrap_df,
    )

    return classical_bootstrap_df, deep_bootstrap_df, tft_bootstrap_df


def run_walk_forward_pipeline() -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    print("===== STEP 6B: WALK-FORWARD DETAIL =====")

    classical_walk_df = None
    deep_walk_df = None
    tft_walk_df = None

    if ROBUSTNESS_INCLUDE_CLASSICAL:
        classical_walk_df = run_classical_walk_forward()

    if ROBUSTNESS_INCLUDE_DEEP:
        deep_walk_df = run_deep_walk_forward(seq_len=DEEP_SEQ_LEN)

    if ROBUSTNESS_INCLUDE_TFT:
        tft_walk_df = run_tft_walk_forward()

    aggregate_walk_forward_details(
        classical_df=classical_walk_df,
        deep_df=deep_walk_df,
        tft_df=tft_walk_df,
    )

    return classical_walk_df, deep_walk_df, tft_walk_df


def run_regime_pipeline() -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    print("===== STEP 6C: REGIME ANALYSIS =====")

    classical_regime_df = None
    deep_regime_df = None
    tft_regime_df = None

    if ROBUSTNESS_INCLUDE_CLASSICAL:
        classical_regime_df = run_classical_regime()

    if ROBUSTNESS_INCLUDE_DEEP:
        deep_regime_df = run_deep_regime(seq_len=DEEP_SEQ_LEN)

    if ROBUSTNESS_INCLUDE_TFT:
        tft_regime_df = run_tft_regime()

    aggregate_regime_details(
        classical_df=classical_regime_df,
        deep_df=deep_regime_df,
        tft_df=tft_regime_df,
    )

    return classical_regime_df, deep_regime_df, tft_regime_df


def run_cross_task_pipeline() -> dict:
    print("===== STEP 7: CROSS-TASK CONSISTENCY =====")
    outputs = run_cross_task_consistency_analysis()
    for name, df in outputs.items():
        if df is not None:
            print(f"{name}: {df.shape}")
    return outputs


def save_master_manifest() -> pd.DataFrame:
    """
    Save a simple manifest of the key output files produced by the full pipeline.
    """
    print("===== STEP 8: SAVING OUTPUT MANIFEST =====")

    key_files = [
        os.path.join(RESULTS_DIR, "model_comparison_expanding.csv"),
        os.path.join(RESULTS_DIR, "deep_model_comparison_expanding.csv"),
        os.path.join(RESULTS_DIR, "tft_model_comparison_expanding.csv"),
        os.path.join(RESULTS_DIR, "all_model_comparison_expanding.csv"),
        os.path.join(ROBUSTNESS_RESULTS_DIR, "bootstrap_detail_all_models.csv"),
        os.path.join(ROBUSTNESS_RESULTS_DIR, "bootstrap_summary_all_models.csv"),
        os.path.join(ROBUSTNESS_RESULTS_DIR, "walk_forward_detail_all_models.csv"),
        os.path.join(ROBUSTNESS_RESULTS_DIR, "regime_analysis_all_models.csv"),
        os.path.join(ROBUSTNESS_RESULTS_DIR, "primary_task_table.csv"),
        os.path.join(ROBUSTNESS_RESULTS_DIR, "standard_summary.csv"),
        os.path.join(ROBUSTNESS_RESULTS_DIR, "bootstrap_summary.csv"),
        os.path.join(ROBUSTNESS_RESULTS_DIR, "regime_summary.csv"),
        os.path.join(ROBUSTNESS_RESULTS_DIR, "final_leaderboard.csv"),
    ]

    rows = []
    for path in key_files:
        rows.append({
            "file_path": path,
            "exists": os.path.exists(path),
            "size_bytes": os.path.getsize(path) if os.path.exists(path) else np.nan,
        })

    manifest_df = pd.DataFrame(rows)
    manifest_path = os.path.join(RESULTS_DIR, "pipeline_output_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Saved output manifest to {manifest_path}")
    return manifest_df


def main():
    ensure_dirs()

    print("===== FULL PIPELINE START =====")
    print_pipeline_summary()

    run_data_preparation()

    if RUN_STANDARD_PIPELINE:
        run_standard_pipeline()

    if RUN_BOOTSTRAP_ROBUSTNESS:
        run_bootstrap_pipeline()

    if RUN_WALK_FORWARD_DETAIL:
        run_walk_forward_pipeline()

    if RUN_REGIME_ANALYSIS:
        run_regime_pipeline()

    if RUN_CROSS_TASK_CONSISTENCY:
        run_cross_task_pipeline()

    # Optional manifest
    try:
        import numpy as np  # local import so manifest helper can use np.nan
        save_master_manifest()
    except Exception as e:
        print(f"[WARN] Failed to save manifest: {e}")

    print("===== FULL PIPELINE COMPLETE =====")


if __name__ == "__main__":
    main()