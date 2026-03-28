# main.py
import os
import shutil
import pandas as pd

from feature_engineering import run_feature_pipeline
from train_eval import (
    run_all_datasets,
    DATASET_CONFIG,
    get_expected_model_names,
    model_exists,
)
from tft_train_eval import run_all_tft_datasets
from deep_train_eval import run_all_deep



PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
MODEL_DIR = "models_saved"
MODEL_DIR_TFT = "models_saved_tft"
MODEL_DIR_DEEP = "models_saved_deep"
BEST_DIR = "models_best"

REQUIRED_DATASETS = list(DATASET_CONFIG.keys())


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
    print("TFT datasets: dataset_return.csv, dataset_volatility.csv")
    print("Deep learning datasets: all 4 datasets")
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


def main():
    print("===== PIPELINE START =====")
    print_pipeline_summary()


    # --------------------------------------------------
    # Step 3: TFT (regression + classification with Platt scaling)
    # --------------------------------------------------
    print("[3/6] Running TFT evaluation / result aggregation...")
    tft_results = run_all_tft_datasets(split_mode="expanding")
    print(tft_results)

    # --------------------------------------------------
    # Step 4: Other deep learning models
    # --------------------------------------------------
    print("[4/6] Running deep learning evaluation / result aggregation...")
    deep_results = run_all_deep()
    print(deep_results)

    # --------------------------------------------------
    # Step 5: Combine all results
    # --------------------------------------------------
    print("[5/6] Combining all model results...")


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
    print(f"Saved combined results to {all_results_path}")



    print("===== PIPELINE COMPLETE =====")


if __name__ == "__main__":
    main()