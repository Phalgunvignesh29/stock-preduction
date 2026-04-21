from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import Config

class GRUPredictor:
    """GRU-based deep learning model for time-series forecasting."""
    
    @staticmethod
    def build_model(input_shape):
        """
        Builds and compiles a 3-layer GRU model. 
        Provides an alternative architecture to LSTM for comparison.
        
        Args:
            input_shape (tuple): Shape of the input data (time_steps, features)
            
        Returns:
            keras.Model: Compiled Keras model
        """
        model = Sequential([
            Input(shape=input_shape),
            
            # Layer 1
            GRU(Config.GRU_UNITS[0], return_sequences=True),
            Dropout(Config.DROPOUT_RATES[0]),
            
            # Layer 2
            GRU(Config.GRU_UNITS[1], return_sequences=True),
            Dropout(Config.DROPOUT_RATES[1]),
            
            # Layer 3
            GRU(Config.GRU_UNITS[2], return_sequences=False),
            Dropout(Config.DROPOUT_RATES[2]),
            
            # Output Layer
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=Config.LEARNING_RATE)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        print("GRU Model Summary:")
        model.summary()
        return model
