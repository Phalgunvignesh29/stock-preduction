import os
from datetime import date

class Config:
    # Project Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'outputs', 'models')
    PLOTS_DIR = os.path.join(BASE_DIR, 'outputs', 'plots')
    SCALERS_DIR = os.path.join(BASE_DIR, 'outputs', 'scalers')
    
    # Data Collection Settings
    DEFAULT_TICKERS = ["AAPL", "NVDA", "RELIANCE.NS"]
    START_DATE = "2018-01-01"
    END_DATE = date.today().strftime("%Y-%m-%d")
    
    # Preprocessing Settings
    LOOKBACK_WINDOW = 60
    TRAIN_TEST_SPLIT = 0.8
    
    # Model Hyperparameters
    LSTM_UNITS = [128, 64, 32]
    GRU_UNITS = [128, 64, 32]
    DROPOUT_RATES = [0.3, 0.25, 0.2]
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    EPOCHS = 50
    PATIENCE = 10
    
    # Ensemble Settings
    ENSEMBLE_WEIGHT_LSTM = 0.6
    ENSEMBLE_WEIGHT_XGB = 0.4
    
    @classmethod
    def setup_directories(cls):
        """Ensure all required output directories exist."""
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.PLOTS_DIR, cls.SCALERS_DIR]:
            os.makedirs(directory, exist_ok=True)

# Run directory setup on import
Config.setup_directories()
