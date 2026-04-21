import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import Config

class ModelEvaluator:
    """Evaluates model predictions and generates visualizations."""
    
    def __init__(self):
        self.plots_dir = Config.PLOTS_DIR
        
    def evaluate(self, y_true, y_pred, model_name="Model"):
        """Calculates regression metrics (RMSE, MAE)."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        print(f"\n--- {model_name} Results ---")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        
        return {"RMSE": rmse, "MAE": mae}
        
    def compare_models(self, results_dict):
        """Prints a comparison table of multiple models."""
        df = pd.DataFrame(results_dict).T
        print("\n--- Model Comparison ---")
        print(df.to_string())
        return df

    def plot_actual_vs_predicted(self, y_true, y_pred, ticker, model_name, dates=None):
        """Plots the actual stock price vs the predicted price."""
        plt.figure(figsize=(14, 7))
        
        if dates is not None:
            plt.plot(dates, y_true, label=f'Actual {ticker} Price', color='blue')
            plt.plot(dates, y_pred, label=f'Predicted {ticker} Price ({model_name})', color='red')
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10)) # Limit x-axis tick density
            plt.xticks(rotation=45)
        else:
            plt.plot(y_true, label=f'Actual {ticker} Price', color='blue')
            plt.plot(y_pred, label=f'Predicted {ticker} Price ({model_name})', color='red')
            
        plt.title(f'{ticker} Stock Price Prediction ({model_name})', fontsize=16)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.plots_dir, f"{ticker}_{model_name}_prediction.png")
        plt.savefig(filepath)
        plt.close()
        print(f"Prediction plot saved to {filepath}")
        
    def plot_training_history(self, history, ticker, model_name):
        """Plots training and validation loss over epochs."""
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss (MSE)')
        plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
        plt.title(f'{model_name.upper()} Model Loss - {ticker}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filepath = os.path.join(self.plots_dir, f"{ticker}_{model_name}_loss.png")
        plt.savefig(filepath)
        plt.close()
        print(f"Loss plot saved to {filepath}")
        
    def plot_sma_chart(self, df, ticker):
        """Plots the stock closing price along with technical indicators (SMA)."""
        plt.figure(figsize=(14, 7))
        plt.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.6)
        
        if 'SMA_20' in df.columns:
            plt.plot(df.index, df['SMA_20'], label='SMA-20', color='orange')
        if 'SMA_50' in df.columns:
            plt.plot(df.index, df['SMA_50'], label='SMA-50', color='green')
            
        plt.title(f'{ticker} Technical Indicators (SMA)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filepath = os.path.join(self.plots_dir, f"{ticker}_technical_indicators.png")
        plt.savefig(filepath)
        plt.close()
        print(f"Technical indicators plot saved to {filepath}")
