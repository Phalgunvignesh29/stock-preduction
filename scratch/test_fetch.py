import yfinance as yf
import pandas as pd

def test(ticker):
    print(f"Testing {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(start="2018-01-01", end="2024-01-01")
    if df.empty:
        print(f"FAILED: {ticker} returned empty")
    else:
        print(f"SUCCESS: {ticker} returned {len(df)} rows")

test("RELIANCE")
test("RELIANCE.NS")
test("AAPL")
