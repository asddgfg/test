import os
import yfinance as yf


DATA_DIR = "data/raw"

TICKERS = [
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "AAPL",
    "MSFT",
    "NVDA",
    "^VIX"
]


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def download_ticker(ticker, start="2010-01-01", end="2025-12-31"):
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)

    if df.empty:
        print(f"Warning: {ticker} returned empty data.")
        return

    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    file_name = ticker.replace("^", "") + ".csv"
    path = os.path.join(DATA_DIR, file_name)

    df.to_csv(path, index=False)
    print(f"Saved to {path}")

def download_all():
    ensure_data_dir()

    for ticker in TICKERS:
        download_ticker(ticker)


if __name__ == "__main__":
    download_all()