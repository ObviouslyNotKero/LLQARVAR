import os

import yfinance as yf

TICKERS = ["BTC-USD", "ETH-USD", "TSLA", "^GSPC"]
FILE_NAMES = {"^GSPC": "SPX"}
START_DATE = "2017-11-09"
END_DATE = None  # None downloads up to today

os.makedirs("data", exist_ok=True)
for ticker in TICKERS:
    prices = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    prices.columns = [column[0] if isinstance(column, tuple) else column for column in prices.columns]
    closes = prices[["Close"]].dropna().reset_index()
    closes.columns = ["Date", "Close"]
    closes.to_csv(f"data/{FILE_NAMES.get(ticker, ticker)}.csv", index=False)
    print(ticker, len(closes), "rows", closes["Date"].min().date(), "to", closes["Date"].max().date())