# robustness_retrain.py
from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd

from train_eval import (
    ROBUSTNESS_RESULTS_DIR as CLASSICAL_ROBUSTNESS_RESULTS_DIR,
    run_bootstrap_all_datasets as run_classical_bootstrap_all,
    run_walk_forward_detail_for_dataset as run_classical_walk_forward_detail_for_dataset,
    summarize_bootstrap_results as summarize_classical_bootstrap_results,
    DATASET_CONFIG as CLASSICAL_DATASET_CONFIG,
)
from deep_train_eval import (
    ROBUSTNESS_RESULTS_DIR as DEEP_ROBUSTNESS_RESULTS_DIR,
    run_deep_bootstrap_all_datasets,
    run_deep_walk_forward_detail_for_dataset,
    summarize_deep_bootstrap_results,
)
from tft_train_eval import (
    ROBUSTNESS_RESULTS_DIR as TFT_ROBUSTNESS_RESULTS_DIR,
    run_tft_bootstrap_all_datasets,
    run_tft_walk_forward_detail_for_dataset,
    summarize_tft_bootstrap_results,
    TFT_DATASET_CONFIG,
)
from regime_analysis import (
    run_classical_regime_analysis_all,
    run_deep_regime_analysis_all,
    run_tft_regime_analysis_all,
)


RESULTS_DIR = "results"
ROBUSTNESS_RESULTS_DIR = os.path.join(RESULTS_DIR, "robustness")


def ensure_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ROBUSTNESS_RESULTS_DIR, exist_ok=True)


def _dedupe_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    valid = [df for df in frames if df is not None and not df.empty]
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, axis=0, ignore_index=True)


def run_classical_bootstrap(
    n_bootstrap: int,
    bootstrap_method: str,
    block_length: int | None,
    base_seed: int,
    save_models: bool,
) -> pd.DataFrame:
    print("\n===== CLASSICAL ML: BOOTSTRAP RETRAINING =====")
    df = run_classical_bootstrap_all(
        split_mode="expanding",
        n_bootstrap=n_bootstrap,
        bootstrap_method=bootstrap_method,
        block_length=block_length,
        base_seed=base_seed,
        save_models=save_models,
    )
    return df


def run_deep_bootstrap(
    n_bootstrap: int,
    bootstrap_method: str,
    block_length: int | None,
    base_seed: int,
    save_models: bool,
    seq_len: int,
) -> pd.DataFrame:
    print("\n===== DEEP LEARNING: BOOTSTRAP RETRAINING =====")
    df = run_deep_bootstrap_all_datasets(
        seq_len=seq_len,
        n_bootstrap=n_bootstrap,
        bootstrap_method=bootstrap_method,
        block_length=block_length,
        base_seed=base_seed,
        save_models=save_models,
    )
    return df


def run_tft_bootstrap(
    n_bootstrap: int,
    bootstrap_method: str,
    block_length: int | None,
    base_seed: int,
    save_models: bool,
) -> pd.DataFrame:
    print("\n===== TFT: BOOTSTRAP RETRAINING =====")
    df = run_tft_bootstrap_all_datasets(
        n_bootstrap=n_bootstrap,
        bootstrap_method=bootstrap_method,
        block_length=block_length,
        base_seed=base_seed,
        save_models=save_models,
    )
    return df


def run_classical_walk_forward() -> pd.DataFrame:
    print("\n===== CLASSICAL ML: WALK-FORWARD DETAIL =====")
    frames = []
    for file_name in CLASSICAL_DATASET_CONFIG:
        df = run_classical_walk_forward_detail_for_dataset(
            file_name=file_name,
            split_mode="expanding",
        )
        frames.append(df)

    out_df = _dedupe_frames(frames)
    if not out_df.empty:
        out_path = os.path.join(ROBUSTNESS_RESULTS_DIR, "classical_walk_forward_detail_all.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Saved combined classical walk-forward detail to {out_path}")
    return out_df


def run_deep_walk_forward(seq_len: int) -> pd.DataFrame:
    print("\n===== DEEP LEARNING: WALK-FORWARD DETAIL =====")
    frames = []
    for file_name in CLASSICAL_DATASET_CONFIG:
        df = run_deep_walk_forward_detail_for_dataset(
            file_name=file_name,
            seq_len=seq_len,
        )
        frames.append(df)

    out_df = _dedupe_frames(frames)
    if not out_df.empty:
        out_path = os.path.join(ROBUSTNESS_RESULTS_DIR, "deep_walk_forward_detail_all.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Saved combined deep walk-forward detail to {out_path}")
    return out_df


def run_tft_walk_forward() -> pd.DataFrame:
    print("\n===== TFT: WALK-FORWARD DETAIL =====")
    frames = []
    for file_name in TFT_DATASET_CONFIG:
        df = run_tft_walk_forward_detail_for_dataset(file_name=file_name)
        frames.append(df)

    out_df = _dedupe_frames(frames)
    if not out_df.empty:
        out_path = os.path.join(ROBUSTNESS_RESULTS_DIR, "tft_walk_forward_detail_all.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Saved combined TFT walk-forward detail to {out_path}")
    return out_df


def run_classical_regime() -> pd.DataFrame:
    print("\n===== CLASSICAL ML: REGIME ANALYSIS =====")
    df = run_classical_regime_analysis_all()
    return df


def run_deep_regime(seq_len: int) -> pd.DataFrame:
    print("\n===== DEEP LEARNING: REGIME ANALYSIS =====")
    df = run_deep_regime_analysis_all(seq_len=seq_len)
    return df


def run_tft_regime() -> pd.DataFrame:
    print("\n===== TFT: REGIME ANALYSIS =====")
    df = run_tft_regime_analysis_all()
    return df


def aggregate_bootstrap_summaries(
    classical_df: pd.DataFrame | None,
    deep_df: pd.DataFrame | None,
    tft_df: pd.DataFrame | None,
) -> pd.DataFrame:
    print("\n===== AGGREGATING BOOTSTRAP SUMMARIES =====")
    summary_frames = []

    if classical_df is not None and not classical_df.empty:
        s = summarize_classical_bootstrap_results(classical_df).copy()
        s["source_group"] = "classical_ml"
        summary_frames.append(s)

    if deep_df is not None and not deep_df.empty:
        s = summarize_deep_bootstrap_results(deep_df).copy()
        s["source_group"] = "deep_learning"
        summary_frames.append(s)

    if tft_df is not None and not tft_df.empty:
        s = summarize_tft_bootstrap_results(tft_df).copy()
        s["source_group"] = "tft"
        summary_frames.append(s)

    final_summary = _dedupe_frames(summary_frames)
    if not final_summary.empty:
        out_path = os.path.join(ROBUSTNESS_RESULTS_DIR, "bootstrap_summary_all_models.csv")
        final_summary.to_csv(out_path, index=False)
        print(f"Saved unified bootstrap summary to {out_path}")

    return final_summary


def aggregate_bootstrap_details(
    classical_df: pd.DataFrame | None,
    deep_df: pd.DataFrame | None,
    tft_df: pd.DataFrame | None,
) -> pd.DataFrame:
    print("\n===== AGGREGATING BOOTSTRAP DETAILS =====")
    final_detail = _dedupe_frames([classical_df, deep_df, tft_df])

    if not final_detail.empty:
        out_path = os.path.join(ROBUSTNESS_RESULTS_DIR, "bootstrap_detail_all_models.csv")
        final_detail.to_csv(out_path, index=False)
        print(f"Saved unified bootstrap detail to {out_path}")

    return final_detail


def aggregate_walk_forward_details(
    classical_df: pd.DataFrame | None,
    deep_df: pd.DataFrame | None,
    tft_df: pd.DataFrame | None,
) -> pd.DataFrame:
    print("\n===== AGGREGATING WALK-FORWARD DETAILS =====")
    final_detail = _dedupe_frames([classical_df, deep_df, tft_df])

    if not final_detail.empty:
        out_path = os.path.join(ROBUSTNESS_RESULTS_DIR, "walk_forward_detail_all_models.csv")
        final_detail.to_csv(out_path, index=False)
        print(f"Saved unified walk-forward detail to {out_path}")

    return final_detail


def aggregate_regime_details(
    classical_df: pd.DataFrame | None,
    deep_df: pd.DataFrame | None,
    tft_df: pd.DataFrame | None,
) -> pd.DataFrame:
    print("\n===== AGGREGATING REGIME ANALYSIS =====")
    final_detail = _dedupe_frames([classical_df, deep_df, tft_df])

    if not final_detail.empty:
        out_path = os.path.join(ROBUSTNESS_RESULTS_DIR, "regime_analysis_all_models.csv")
        final_detail.to_csv(out_path, index=False)
        print(f"Saved unified regime analysis to {out_path}")

    return final_detail


def print_run_header(args) -> None:
    print("===================================================")
    print("ROBUSTNESS RETRAIN ORCHESTRATOR")
    print("===================================================")
    print(f"Run bootstrap:      {args.run_bootstrap}")
    print(f"Run walk-forward:   {args.run_walk_forward}")
    print(f"Run regime:         {args.run_regime}")
    print(f"Run classical:      {args.include_classical}")
    print(f"Run deep:           {args.include_deep}")
    print(f"Run TFT:            {args.include_tft}")
    print(f"n_bootstrap:        {args.n_bootstrap}")
    print(f"bootstrap_method:   {args.bootstrap_method}")
    print(f"block_length:       {args.block_length}")
    print(f"base_seed:          {args.base_seed}")
    print(f"save_models:        {args.save_models}")
    print(f"seq_len (deep):     {args.seq_len}")
    print("===================================================")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run robustness retraining / walk-forward / regime analysis for classical, deep, and TFT models."
    )

    parser.add_argument("--run_bootstrap", action="store_true", help="Run bootstrap retraining.")
    parser.add_argument("--run_walk_forward", action="store_true", help="Run walk-forward detail export.")
    parser.add_argument("--run_regime", action="store_true", help="Run regime analysis.")

    parser.add_argument("--include_classical", action="store_true", help="Include classical ML models.")
    parser.add_argument("--include_deep", action="store_true", help="Include deep learning models.")
    parser.add_argument("--include_tft", action="store_true", help="Include TFT models.")

    parser.add_argument("--n_bootstrap", type=int, default=30, help="Number of bootstrap retraining replications.")
    parser.add_argument(
        "--bootstrap_method",
        type=str,
        default="stationary",
        choices=["stationary", "moving", "circular"],
        help="Bootstrap method.",
    )
    parser.add_argument(
        "--block_length",
        type=int,
        default=None,
        help="Block length. If omitted, each submodule will auto-suggest one.",
    )
    parser.add_argument("--base_seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--save_models", action="store_true", help="Save all bootstrap-trained models.")
    parser.add_argument("--seq_len", type=int, default=20, help="Sequence length for deep learning models.")

    return parser.parse_args()


def main():
    ensure_dirs()
    args = parse_args()

    if not args.run_bootstrap and not args.run_walk_forward and not args.run_regime:
        raise ValueError("You must specify at least one of --run_bootstrap, --run_walk_forward, or --run_regime.")

    if not args.include_classical and not args.include_deep and not args.include_tft:
        raise ValueError("You must specify at least one of --include_classical, --include_deep, --include_tft.")

    print_run_header(args)

    classical_bootstrap_df = None
    deep_bootstrap_df = None
    tft_bootstrap_df = None

    classical_walk_df = None
    deep_walk_df = None
    tft_walk_df = None

    classical_regime_df = None
    deep_regime_df = None
    tft_regime_df = None

    if args.run_bootstrap:
        if args.include_classical:
            classical_bootstrap_df = run_classical_bootstrap(
                n_bootstrap=args.n_bootstrap,
                bootstrap_method=args.bootstrap_method,
                block_length=args.block_length,
                base_seed=args.base_seed,
                save_models=args.save_models,
            )

        if args.include_deep:
            deep_bootstrap_df = run_deep_bootstrap(
                n_bootstrap=args.n_bootstrap,
                bootstrap_method=args.bootstrap_method,
                block_length=args.block_length,
                base_seed=args.base_seed,
                save_models=args.save_models,
                seq_len=args.seq_len,
            )

        if args.include_tft:
            tft_bootstrap_df = run_tft_bootstrap(
                n_bootstrap=args.n_bootstrap,
                bootstrap_method=args.bootstrap_method,
                block_length=args.block_length,
                base_seed=args.base_seed,
                save_models=args.save_models,
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

    if args.run_walk_forward:
        if args.include_classical:
            classical_walk_df = run_classical_walk_forward()

        if args.include_deep:
            deep_walk_df = run_deep_walk_forward(seq_len=args.seq_len)

        if args.include_tft:
            tft_walk_df = run_tft_walk_forward()

        aggregate_walk_forward_details(
            classical_df=classical_walk_df,
            deep_df=deep_walk_df,
            tft_df=tft_walk_df,
        )

    if args.run_regime:
        if args.include_classical:
            classical_regime_df = run_classical_regime()

        if args.include_deep:
            deep_regime_df = run_deep_regime(seq_len=args.seq_len)

        if args.include_tft:
            tft_regime_df = run_tft_regime()

        aggregate_regime_details(
            classical_df=classical_regime_df,
            deep_df=deep_regime_df,
            tft_df=tft_regime_df,
        )

    print("\nAll requested robustness jobs finished.")


if __name__ == "__main__":
    main()