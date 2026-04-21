import xgboost as xgb
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import Config
from src.models.lstm_model import LSTMPredictor

class EnsemblePredictor:
    """
    Ensemble model combining an LSTM and XGBoost.
    LSTM captures sequential patterns, while XGBoost captures non-linear tabular feature interactions.
    """
    
    def __init__(self, lstm_model=None, lstm_input_shape=None):
        """
        Initializes the ensemble. Either requires a pre-built lstm_model or the input_shape to build one.
        """
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=5, 
            random_state=42,
            objective="reg:squarederror"
        )
        
        if lstm_model is not None:
            self.lstm_model = lstm_model
        elif lstm_input_shape is not None:
            self.lstm_model = LSTMPredictor.build_model(lstm_input_shape)
        else:
            self.lstm_model = None
            
        self.lstm_weight = Config.ENSEMBLE_WEIGHT_LSTM
        self.xgb_weight = Config.ENSEMBLE_WEIGHT_XGB

    def _prepare_xgb_data(self, X_seq):
        """
        Flattens 3D sequential data (samples, timesteps, features) 
        into 2D (samples, features) for XGBoost.
        We typically use the features from the LAST time step of the sequence.
        """
        # Take the last time step for all samples across all features
        return X_seq[:, -1, :]

    def train_xgb(self, X_train_seq, y_train):
        """Trains just the XGBoost portion of the ensemble."""
        X_train_flat = self._prepare_xgb_data(X_train_seq)
        print("Training XGBoost Regressor...")
        self.xgb_model.fit(X_train_flat, y_train)
        print("XGBoost training complete.")

    def predict(self, X_test_seq):
        """
        Predicts using both models and performs a weighted average.
        Assumes LSTM is already trained.
        """
        if self.lstm_model is None:
            raise ValueError("LSTM model is not instantiated.")
            
        # 1. Get LSTM Predictions
        lstm_preds = self.lstm_model.predict(X_test_seq, verbose=0).flatten()
        
        # 2. Get XGB Predictions
        X_test_flat = self._prepare_xgb_data(X_test_seq)
        xgb_preds = self.xgb_model.predict(X_test_flat)
        
        # 3. Weighted Average Ensemble
        ensemble_preds = (lstm_preds * self.lstm_weight) + (xgb_preds * self.xgb_weight)
        
        return ensemble_preds, lstm_preds, xgb_preds
