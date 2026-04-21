import argparse
import os
import pandas as pd

from config.settings import Config
from src.data_collector import StockDataCollector
from src.feature_engineer import FeatureEngineer
from src.preprocessor import DataPreprocessor
from src.models.lstm_model import LSTMPredictor
from src.models.gru_model import GRUPredictor
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.sentiment.sentiment_analyzer import NewsSentimentAnalyzer

def run_pipeline(ticker, mode='train', model_type='lstm'):
    """
    Executes the full pipeline for training or evaluation.
    """
    print(f"=== Running Stock Prediction Pipeline for {ticker} ({model_type.upper()}) ===")
    
    # 1. Data Collection
    collector = StockDataCollector()
    df = collector.fetch_stock(ticker)
    if df is None:
        return
        
    # 2. Feature Engineering & Sentiment
    engineer = FeatureEngineer()
    df = engineer.add_all_features(df)
    
    sentiment_analyzer = NewsSentimentAnalyzer()
    df = sentiment_analyzer.get_simulated_historical_sentiment(df)
    
    # 3. Preprocessing
    preprocessor = DataPreprocessor()
    train_df, test_df = preprocessor.chronological_split(df)
    
    # Needs purely numerical data for scaling
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    train_df = train_df[num_cols]
    test_df = test_df[num_cols]
    
    if mode == 'train':
        scaled_train, scaled_test, features = preprocessor.transform(train_df, test_df)
        preprocessor.save_scaler(ticker) # Save scaler so it can be used for eval/prediction later
    else: # evaluate
        if not preprocessor.load_scaler(ticker):
            print("Please 'train' first to generate scaler.")
            return
        # During evaluation, just apply transform based on loaded scaler settings
        # Caution: preprocessor.transform fits the scaler again.
        # So we manually apply the loaded scaler.
        # Find target column index:
        cols = list(train_df.columns)
        if 'Close' in cols:
            cols.remove('Close')
            cols = ['Close'] + cols
        
        train_df = train_df[cols]
        test_df = test_df[cols]
        preprocessor.target_col_idx = 0
        
        scaled_train = preprocessor.scaler.transform(train_df)
        scaled_test = preprocessor.scaler.transform(test_df)
        features = cols

    X_train, y_train = preprocessor.create_sequences(scaled_train)
    X_test, y_test = preprocessor.create_sequences(scaled_test)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # 4. Model Building / Loading
    if model_type == 'lstm':
        model = LSTMPredictor.build_model(input_shape)
    else:
        model = GRUPredictor.build_model(input_shape)
        
    trainer = ModelTrainer(model_name=model_type)
    evaluator = ModelEvaluator()
    
    model_path = os.path.join(Config.MODELS_DIR, f"{ticker}_{model_type}_best.keras")
    
    if mode == 'train':
        history = trainer.train(model, X_train, y_train, X_test, y_test, ticker)
        evaluator.plot_training_history(history, ticker, model_type)
        print(f"Model saved to {model_path}")
        
    elif mode == 'evaluate':
        if os.path.exists(model_path):
            model.load_weights(model_path)
            print(f"Loaded weights from {model_path}")
        else:
            print(f"Model weights not found at {model_path}. Please train first.")
            return

        # 5. Evaluation Phase
        # Get scaled predictions
        y_pred_scaled = model.predict(X_test)
        
        # Inverse transform to original price scale
        y_test_original = preprocessor.inverse_transform_target(y_test.reshape(-1, 1))
        y_pred_original = preprocessor.inverse_transform_target(y_pred_scaled)
        
        # Calculate metrics
        results = evaluator.evaluate(y_test_original, y_pred_original, model_name=model_type.upper())
        
        # Get test dates corresponding to the sequences
        test_dates = test_df.index[Config.LOOKBACK_WINDOW:]
        evaluator.plot_actual_vs_predicted(y_test_original, y_pred_original, ticker, model_type.upper(), test_dates)
        evaluator.plot_sma_chart(df, ticker)
        
        print("\nPipeline execution complete. Check 'outputs/plots/' for graphs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Market Prediction Pipeline")
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], default='train', help='Run mode')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock Ticker Symbol')
    parser.add_argument('--model', type=str, choices=['lstm', 'gru'], default='lstm', help='Model Architecture')
    
    args = parser.parse_args()
    
    run_pipeline(ticker=args.ticker, mode=args.mode, model_type=args.model)
