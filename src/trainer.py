import os
import sys
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import Config

class ModelTrainer:
    """Handles the training loop for Keras deep learning models."""
    
    def __init__(self, model_name="lstm"):
        self.model_name = model_name
        self.models_dir = Config.MODELS_DIR
        
    def get_callbacks(self, ticker):
        """Sets up ModelCheckpoint and EarlyStopping callbacks."""
        filepath = os.path.join(self.models_dir, f"{ticker}_{self.model_name}_best.keras")
        
        # Stop training when a monitored metric has stopped improving
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=Config.PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        
        # Save the model after every epoch (if it's the best)
        checkpoint = ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        return [early_stop, checkpoint]
        
    def train(self, model, X_train, y_train, X_val, y_val, ticker, 
              epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE):
        """
        Executes the training loop.
        
        Args:
            model (keras.Model): Compiled model
            X_train, y_train: Training data
            X_val, y_val: Validation data
            ticker (str): Stock ticker for saving files
            epochs (int): Number of epochs
            batch_size (int): Batch size
            
        Returns:
            history: Keras training history object
        """
        print(f"\n--- Starting Training for {ticker} ({self.model_name.upper()}) ---")
        callbacks = self.get_callbacks(ticker)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining Complete.")
        return history
