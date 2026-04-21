import pandas as pd
import numpy as np

class FeatureEngineer:
    """Class to add technical indicators (features) to stock financial data."""
    
    @staticmethod
    def add_sma(df, window):
        """
        Adds Simple Moving Average (SMA).
        SMA smooths out price data to identify trends over a specific window.
        """
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        return df
        
    @staticmethod
    def add_rsi(df, period=14):
        """
        Adds Relative Strength Index (RSI).
        RSI measures the magnitude of recent price changes to evaluate overbought or oversold conditions.
        """
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Prevent division by zero
        rs = gain / loss.replace(0, np.nan)
        rs = rs.fillna(0) # If loss is 0, RS is effectively infinite. For RSI formulation, infinite RS -> RSI = 100
        
        # Calculate real RSI where loss is zero
        df['RSI'] = 100 - (100 / (1 + rs))
        df.loc[loss == 0, 'RSI'] = 100 
        return df
        
    @staticmethod
    def add_macd(df, fast=12, slow=26, signal=9):
        """
        Adds Moving Average Convergence Divergence (MACD).
        MACD is a trend-following momentum indicator showing relationship between two EMAs.
        """
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        
        df['MACD'] = ema_fast - ema_slow
        df['Signal_Line'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        return df

    @staticmethod
    def add_volatility(df, window=20):
        """Adds standard deviation of returns as a proxy for volatility."""
        df['Returns'] = df['Close'].pct_change()
        df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std()
        return df

    @classmethod
    def add_all_features(cls, df):
        """
        Orchestrator to add all predefined technical features to the dataset.
        Drops NaN values created by rolling windows at the beginning of the dataframe.
        """
        df = df.copy()
        
        # Add indicators
        df = cls.add_sma(df, window=20)
        df = cls.add_sma(df, window=50)
        df = cls.add_rsi(df, period=14)
        df = cls.add_macd(df)
        df = cls.add_volatility(df, window=20)
        
        # Drop rows with NaN values resulting from moving averages and diffs
        df.dropna(inplace=True)
        return df

# Example testing block
if __name__ == "__main__":
    from data_collector import StockDataCollector
    collector = StockDataCollector()
    df = collector.fetch_stock("AAPL")
    if df is not None:
        engineer = FeatureEngineer()
        df_engineered = engineer.add_all_features(df)
        print(f"Shape before engineering: {df.shape}")
        print(f"Shape after engineering (NaNs dropped): {df_engineered.shape}")
        print(df_engineered[['Close', 'SMA_20', 'RSI', 'MACD']].tail())
