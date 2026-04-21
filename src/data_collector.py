import yfinance as yf
import pandas as pd
import os
import sys
import contextlib
import io

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import Config

class StockDataCollector:
    """Class to fetch and handle stock data using yfinance."""
    
    def __init__(self):
        self.data_dir = Config.DATA_DIR
        
    def fetch_stock(self, ticker, start_date=Config.START_DATE, end_date=Config.END_DATE):
        """
        Fetches historical stock data from Yahoo Finance.
        Automatically tries .NS (NSE) and .BO (BSE) suffixes for Indian stocks
        if the bare ticker symbol returns no data.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL', 'RELIANCE', 'RELIANCE.NS')
            start_date (str): Start date in YYYY-MM-DD
            end_date (str): End date in YYYY-MM-DD
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        # Build list of tickers to try: original first, then with Indian exchange suffixes
        tickers_to_try = [ticker]
        if '.' not in ticker:
            tickers_to_try += [ticker + '.NS', ticker + '.BO']

        for t in tickers_to_try:
            print(f"Fetching data for {t} from {start_date} to {end_date}...")
            try:
                stock = yf.Ticker(t)
                
                # Suppress yfinance console spam ("possibly delisted" warnings)
                f = io.StringIO()
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    df = stock.history(start=start_date, end=end_date, auto_adjust=True)
                
                if df.empty:
                    print(f"No data for {t}, trying next...")
                    continue
                    
                # Drop timezone information from index if present
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(None)
                    
                print(f"Successfully fetched {len(df)} rows for {t}.")
                # Store the resolved ticker so callers can know what was used
                df.attrs['resolved_ticker'] = t
                return df
            except Exception as e:
                print(f"Error fetching data for {t}: {e}")
                continue
        
        print(f"Warning: No data found for {ticker} (tried {tickers_to_try}).")
        return None
            
    def fetch_multiple(self, tickers=Config.DEFAULT_TICKERS, start_date=Config.START_DATE, end_date=Config.END_DATE):
        """
        Fetches data for multiple tickers.
        
        Returns:
            dict: A dictionary mapping tickers to their respective DataFrames.
        """
        data_dict = {}
        for ticker in tickers:
            df = self.fetch_stock(ticker, start_date, end_date)
            if df is not None:
                data_dict[ticker] = df
        return data_dict

    def save_to_csv(self, df, ticker, filename=None):
        """Saves a DataFrame to CSV in the data directory."""
        if filename is None:
            filename = f"{ticker}_historical.csv"
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath)
        print(f"Data saved to {filepath}")
        return filepath
        
    def load_from_csv(self, ticker, filename=None):
        """Loads a DataFrame from CSV located in the data directory."""
        if filename is None:
            filename = f"{ticker}_historical.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
            print(f"Data loaded from {filepath}")
            return df
        else:
            print(f"File {filepath} not found.")
            return None

# Example testing block
if __name__ == "__main__":
    collector = StockDataCollector()
    df = collector.fetch_stock("AAPL")
    if df is not None:
        print(df.head())
        collector.save_to_csv(df, "AAPL")
