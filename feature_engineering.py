import os
from typing import Dict, List

import numpy as np
import pandas as pd


RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


ASSETS = {
    "SPY": "spy",
    "QQQ": "qqq",
    "DIA": "dia",
    "IWM": "iwm",
    "AAPL": "aapl",
    "MSFT": "msft",
    "NVDA": "nvda",
    "VIX": "vix",
}


def ensure_processed_dir() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_single_asset_csv(symbol: str, prefix: str) -> pd.DataFrame:
    """
    Load a single raw CSV from data/raw and standardize column names.

    Expected columns from yfinance CSV:
    Date, Open, High, Low, Close, Adj Close, Volume
    """
    path = os.path.join(RAW_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing raw data file: {path}")

    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"'Date' column not found in {path}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    rename_map = {}
    for col in df.columns:
        if col == "Date":
            continue
        safe_col = col.lower().replace(" ", "_")
        rename_map[col] = f"{prefix}_{safe_col}"

    df = df.rename(columns=rename_map)
    return df


def load_all_assets() -> pd.DataFrame:
    """
    Load all asset CSVs and merge them on Date using inner join.
    """
    merged = None

    for symbol, prefix in ASSETS.items():
        asset_df = load_single_asset_csv(symbol, prefix)
        if merged is None:
            merged = asset_df
        else:
            merged = merged.merge(asset_df, on="Date", how="inner")

    if merged is None:
        raise ValueError("No asset data loaded.")

    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged


def _safe_return(series: pd.Series, periods: int) -> pd.Series:
    return series.pct_change(periods=periods)


def _rolling_vol(series: pd.Series, window: int) -> pd.Series:
    return series.pct_change().rolling(window).std()


def _ma_gap(series: pd.Series, window: int) -> pd.Series:
    ma = series.rolling(window).mean()
    return series / ma - 1.0


def _drawdown(series: pd.Series, window: int) -> pd.Series:
    rolling_max = series.rolling(window).max()
    return series / rolling_max - 1.0


def add_spy_features(df: pd.DataFrame) -> pd.DataFrame:
    price = df["spy_close"]
    volume = df["spy_volume"]

    df["spy_ret_1"] = _safe_return(price, 1)
    df["spy_ret_5"] = _safe_return(price, 5)
    df["spy_ret_10"] = _safe_return(price, 10)
    df["spy_ret_20"] = _safe_return(price, 20)

    df["spy_ma_gap_5"] = _ma_gap(price, 5)
    df["spy_ma_gap_10"] = _ma_gap(price, 10)
    df["spy_ma_gap_20"] = _ma_gap(price, 20)
    df["spy_ma_gap_50"] = _ma_gap(price, 50)

    df["spy_vol_5"] = _rolling_vol(price, 5)
    df["spy_vol_10"] = _rolling_vol(price, 10)
    df["spy_vol_20"] = _rolling_vol(price, 20)

    df["spy_hl_spread"] = (df["spy_high"] - df["spy_low"]) / df["spy_close"]
    df["spy_oc_ret"] = (df["spy_close"] - df["spy_open"]) / df["spy_open"]
    df["spy_volume_chg"] = volume.pct_change()

    df["spy_drawdown_20"] = _drawdown(price, 20)
    return df


def add_secondary_etf_features(df: pd.DataFrame) -> pd.DataFrame:
    for prefix in ["qqq", "dia", "iwm"]:
        price = df[f"{prefix}_close"]

        df[f"{prefix}_ret_1"] = _safe_return(price, 1)
        df[f"{prefix}_ret_5"] = _safe_return(price, 5)
        df[f"{prefix}_vol_10"] = _rolling_vol(price, 10)
        df[f"{prefix}_ma_gap_20"] = _ma_gap(price, 20)

    return df


def add_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    for prefix in ["aapl", "msft", "nvda"]:
        price = df[f"{prefix}_close"]

        df[f"{prefix}_ret_1"] = _safe_return(price, 1)
        df[f"{prefix}_ret_5"] = _safe_return(price, 5)
        df[f"{prefix}_vol_10"] = _rolling_vol(price, 10)

    return df


def add_vix_features(df: pd.DataFrame) -> pd.DataFrame:
    level = df["vix_close"]

    df["vix_ret_1"] = level.pct_change()
    df["vix_level"] = level
    df["vix_ma_gap_10"] = _ma_gap(level, 10)
    return df


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_spy_features(df)
    df = add_secondary_etf_features(df)
    df = add_stock_features(df)
    df = add_vix_features(df)
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 4 targets aligned with the proposal:

    1) y_return: next-day SPY return
    2) y_direction: sign of next-day SPY return
    3) y_vol: future 5-day realized volatility
    4) y_regime: future 20-day high/low volatility regime
    """
    spy_ret_1 = df["spy_close"].pct_change()

    # 1. Return prediction target
    df["y_return"] = np.log(df["spy_close"].shift(-1) / df["spy_close"])

    # 2. Direction classification target
    df["y_direction"] = (df["y_return"] > 0).astype(int)

    # 3. Future 5-day realized volatility target
    future_5d_vol = (
        spy_ret_1.shift(-1)
        .rolling(window=5)
        .std()
        .shift(-4)
    )
    df["y_vol"] = future_5d_vol

    # 4. Future 20-day volatility regime target
    future_20d_vol = (
        spy_ret_1.shift(-1)
        .rolling(window=20)
        .std()
        .shift(-19)
    )
    df["future_20d_vol"] = future_20d_vol

    rolling_median = df["future_20d_vol"].rolling(window=252, min_periods=60).median()
    df["y_regime"] = (df["future_20d_vol"] > rolling_median).astype(int)

    return df


def get_feature_columns() -> List[str]:
    return [
        "spy_ret_1",
        "spy_ret_5",
        "spy_ret_10",
        "spy_ret_20",
        "spy_ma_gap_5",
        "spy_ma_gap_10",
        "spy_ma_gap_20",
        "spy_ma_gap_50",
        "spy_vol_5",
        "spy_vol_10",
        "spy_vol_20",
        "spy_hl_spread",
        "spy_oc_ret",
        "spy_volume_chg",
        "spy_drawdown_20",
        "qqq_ret_1",
        "qqq_ret_5",
        "qqq_vol_10",
        "qqq_ma_gap_20",
        "dia_ret_1",
        "dia_ret_5",
        "dia_vol_10",
        "dia_ma_gap_20",
        "iwm_ret_1",
        "iwm_ret_5",
        "iwm_vol_10",
        "iwm_ma_gap_20",
        "aapl_ret_1",
        "aapl_ret_5",
        "aapl_vol_10",
        "msft_ret_1",
        "msft_ret_5",
        "msft_vol_10",
        "nvda_ret_1",
        "nvda_ret_5",
        "nvda_vol_10",
        "vix_ret_1",
        "vix_level",
        "vix_ma_gap_10",
    ]


def build_master_dataset() -> pd.DataFrame:
    df = load_all_assets()
    df = add_all_features(df)
    df = add_targets(df)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows with fully available features and labels.
    """
    required_cols = get_feature_columns() + [
        "y_return",
        "y_direction",
        "y_vol",
        "y_regime",
    ]
    cleaned = df.dropna(subset=required_cols).reset_index(drop=True)
    return cleaned


def save_outputs(df: pd.DataFrame) -> Dict[str, str]:
    """
    Save:
    - master_dataset.csv
    - dataset_return.csv
    - dataset_direction.csv
    - dataset_volatility.csv
    - dataset_regime.csv
    """
    ensure_processed_dir()

    paths = {}

    master_path = os.path.join(PROCESSED_DIR, "master_dataset.csv")
    df.to_csv(master_path, index=False)
    paths["master"] = master_path

    feature_cols = ["Date"] + get_feature_columns()

    datasets = {
        "dataset_return.csv": feature_cols + ["y_return"],
        "dataset_direction.csv": feature_cols + ["y_direction"],
        "dataset_volatility.csv": feature_cols + ["y_vol"],
        "dataset_regime.csv": feature_cols + ["y_regime"],
    }

    for file_name, cols in datasets.items():
        out_path = os.path.join(PROCESSED_DIR, file_name)
        df[cols].to_csv(out_path, index=False)
        paths[file_name] = out_path

    return paths


def run_feature_pipeline() -> Dict[str, str]:
    df = build_master_dataset()
    df = clean_dataset(df)
    paths = save_outputs(df)
    return paths


if __name__ == "__main__":
    output_paths = run_feature_pipeline()
    print("Saved processed datasets:")
    for name, path in output_paths.items():
        print(f"  {name}: {path}")