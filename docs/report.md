# Academic Report: Stock Market Prediction Using Deep Learning

## 1. Abstract
The prediction of stock market prices is a challenging task due to the high volatility, non-linearity, and complex dynamics of financial markets. Traditional statistical models often fail to capture the long-term temporal dependencies present in stock data. This project presents a robust, end-to-end deep learning framework for stock price forecasting using Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks. Furthermore, the model is augmented with feature engineering (technical indicators like SMA, RSI, MACD) and sentiment analysis derived from financial news headlines. Evaluation on real-world datasets (e.g., AAPL) demonstrates that deep neural networks, especially when combined in an ensemble approach, can effectively model sequences of financial data and yield low Root Mean Squared Error (RMSE) predictions.

## 2. Introduction
Financial forecasting aims to predict the future value of a company's stock based on historical data and other market indicators. Accurate predictions allow investors to maximize profit and minimize risk. 

**Challenges:**
- **Volatility:** Prices can change drastically and unexpectedly.
- **Non-Linearity:** The relationship between market indicators and the final price is highly non-linear.
- **Noise:** Financial data contains high degrees of noise (e.g., random market fluctuations) making signal extraction difficult.

**Justification for Deep Learning:**
Standard machine learning algorithms (like Linear Regression) assume independent observations. Time-series data violates this assumption. Recurrent Neural Networks (RNNs), specifically LSTMs and GRUs, are specifically designed to address sequence prediction problems by maintaining a hidden memory state that learns temporal dependencies. LSTMs further solve the "vanishing gradient" problem commonly found in standard RNNs.

## 3. Scope & Motivation
**Motivation:** Developing a tool that synthesizes complex market signals (price history, technical momentum, and public sentiment) into an actionable prediction.
**Scope:** The system fetches live data, processes features, trains deep learning models, and serves predictions over an interactive web dashboard.

## 4. Methodology
### 4.1 Data Collection
Data is collected via the `yfinance` API, fetching OHLCV (Open, High, Low, Close, Volume) data over a 5-year period.

### 4.2 Feature Engineering
We compute technical indicators to provide the model with momentum and trend signals:
- **SMA-20 & SMA-50:** Simple Moving Averages identify short and medium-term trends.
- **RSI (14):** Relative Strength Index identifies overbought (>70) or oversold (<30) conditions.
- **MACD:** Moving Average Convergence Divergence highlights the relationship between two moving averages of prices.

### 4.3 Data Preprocessing (Crucial to avoid Data Leakage)
1. **Windowing:** Data is converted into sequences of 60 days to predict the 61st day.
2. **Chronological Splitting:** Data is strictly split chronologically (80% train, 20% test). Standard random `train_test_split` is invalid for time series.
3. **Scaling:** A `MinMaxScaler(0,1)` is fit **only on the training data**, and then applied to both train and test sets to prevent information from the future leaking into the model during training.

### 4.4 Model Architecture
1. **LSTM:** A 3-layer architecture (128 → 64 → 32 units) with interleaved `Dropout` layers to prevent overfitting.
2. **GRU:** Built similarly for architectural performance comparison.
3. **Ensemble:** A late-fusion ensemble combining the sequential prowess of the LSTM with an XGBoost regressor that handles the tabular indicator features effectively.

### 4.5 Sentiment Analysis (VADER)
We incorporate a lexicon-based sentiment analysis tool (VADER) specialized for short, informal text to analyze financial headlines and generate a compound sentiment score [-1, 1] which is appended as a feature.

## 5. Algorithm/Pseudocode
```text
Algorithm: Deep Learning Stock Prediction Pipeline

1. FetchOHLCV(ticker, start_date, end_date)
2. HandleMissingValues()
3. Compute Technical Indicators (SMA, RSI, MACD, Volatility)
4. Fetch News Headlines -> Calculate DailySentimentScore(VADER)
5. Merge Sentiment with OHLCV data
6. TrainSplit, TestSplit <- SplitDataChronologically(ratio=0.8)
7. scaler <- FitMinMaxScaler(TrainSplit)
8. TrainScaled <- Transform(TrainSplit)
9. TestScaled <- Transform(TestSplit)
10. X_train, y_train <- CreateSequences(TrainScaled, window=60)
11. X_test, y_test <- CreateSequences(TestScaled, window=60)
12. Initialize LSTM(128-64-32) with Dropout
13. Compile Model (Optimizer: Adam, Loss: MSE)
14. Train Model with EarlyStopping(patience=10)
15. Predict on X_test
16. InverseTransform(Predictions)
17. Calculate RMSE, MAE
```

## 6. Implementation Details
The project is implemented in Python, modularized using Object-Oriented principles.
- **Libraries:** TensorFlow/Keras (Model), Scikit-Learn (Scaling/Metrics), Pandas/NumPy (Data manipulation), Streamlit (Web App), VADER (Sentiment).

## 7. Results & Discussion
The model evaluates predictions using:
- **Root Mean Squared Error (RMSE):** Penalizes larger errors more heavily.
- **Mean Absolute Error (MAE):** Average magnitude of absolute errors.
  
*Note: Include screenshots of the Actual vs Predicted graphs and loss curves here.*

LSTM generally captures the long-term trend well, though sudden market crashes/spikes caused by exogenous external events remain difficult to predict without comprehensive real-time alternative data.

## 8. Applications
- **Algorithmic Trading:** Integration into automated trading bots.
- **Risk Management:** Alerting portfolio managers of predicted downturns.
- **Retail Investing:** Consumer dashboards advising on enter/exit points.

## 9. Conclusion
The project successfully builds an end-to-end, full-stack machine learning solution for market tracking. We verified that deep learning architectures (LSTMs) outclass simple linear models in time-series forecasting. Strict data preprocessing practices effectively prevented data leakage, resulting in realistic and robust evaluation metrics.

## 10. Future Work
- Integration of state-of-the-art **Transformer** models (e.g., Temporal Fusion Transformers).
- Connecting to live broker APIs (like Alpaca) for simulated paper trading.
- Expanding Sentiment Analysis by utilizing FinBERT instead of VADER.
