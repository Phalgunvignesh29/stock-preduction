import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import Config

class DataPreprocessor:
    """Class to scale data and create sequences for sequential models."""
    
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        # Keep track of target column index (usually 'Close' or 'Adj Close')
        self.target_col_idx = 0 
        self.scalers_dir = Config.SCALERS_DIR

    def chronological_split(self, df, ratio=Config.TRAIN_TEST_SPLIT):
        """Splits the dataframe chronologically to prevent data leakage."""
        train_size = int(len(df) * ratio)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        return train_df, test_df

    def transform(self, train_df, test_df, target_col='Close'):
        """
        Fits scaler strictly on training data, then transforms both train and test.
        Returns scaled numpy arrays.
        """
        # Ensure target_col is the first column for easier inverse transform
        cols = list(train_df.columns)
        if target_col in cols:
            cols.remove(target_col)
            cols = [target_col] + cols
            
        train_df = train_df[cols]
        test_df = test_df[cols]
        self.target_col_idx = cols.index(target_col)
        
        # Fit ONLY on train data to prevent data leakage
        scaled_train = self.scaler.fit_transform(train_df)
        scaled_test = self.scaler.transform(test_df)
        
        return scaled_train, scaled_test, cols

    def create_sequences(self, data, lookback=Config.LOOKBACK_WINDOW):
        """
        Converts 2D scaled data into 3D sequences for LSTM/GRU.
        
        Args:
            data (np.array): Scaled data (samples, features)
            lookback (int): Number of previous time steps to use as input variables
            
        Returns:
            np.array, np.array: X (samples, lookback, features), y (samples,)
        """
        X, y = [], []
        # Target variable is assumed to be at target_col_idx (0)
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i, self.target_col_idx])
            
        return np.array(X), np.array(y)
        
    def inverse_transform_target(self, scaled_predictions):
        """
        Inverse transforms the predictions back to their original scale (price scale).
        """
        # Create a dummy array with the same shape as the scaler expected during fit
        num_features = self.scaler.n_features_in_
        dummy = np.zeros((len(scaled_predictions), num_features))
        
        # Insert our predictions into the target column position
        dummy[:, self.target_col_idx] = scaled_predictions.flatten()
        
        # Inverse transform and extract the target column
        inv_scaled = self.scaler.inverse_transform(dummy)
        return inv_scaled[:, self.target_col_idx]

    def save_scaler(self, ticker):
        """Saves the fitted scaler for later inference."""
        filepath = os.path.join(self.scalers_dir, f"{ticker}_scaler.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved down directly to {filepath}")
        
    def load_scaler(self, ticker):
        """Loads a previously fitted scaler."""
        filepath = os.path.join(self.scalers_dir, f"{ticker}_scaler.pkl")
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Scaler loaded from {filepath}")
            return True
        else:
            print(f"Scaler not found at {filepath}")
            return False

# Example block
if __name__ == "__main__":
    from data_collector import StockDataCollector
    from feature_engineer import FeatureEngineer
    
    collector = StockDataCollector()
    df = collector.fetch_stock("AAPL")
    if df is not None:
        engineered_df = FeatureEngineer.add_all_features(df)
        
        preprocessor = DataPreprocessor()
        train_df, test_df = preprocessor.chronological_split(engineered_df)
        
        # Exclude string/categorical columns if any (shouldn't be any in standard OHLCV)
        train_df = train_df.select_dtypes(include=[np.number])
        test_df = test_df.select_dtypes(include=[np.number])
        
        scaled_train, scaled_test, features = preprocessor.transform(train_df, test_df)
        print(f"Scaled Train Shape: {scaled_train.shape}, Features used: {len(features)}")
        
        X_train, y_train = preprocessor.create_sequences(scaled_train)
        X_test, y_test = preprocessor.create_sequences(scaled_test)
        
        print(f"X_train shape (LSTM input): {X_train.shape}")
        print(f"y_train shape (Target): {y_train.shape}")
