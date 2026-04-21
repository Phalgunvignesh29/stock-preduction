# Stock Market Prediction Using Deep Learning

This is a complete, production-level Machine Learning project developed for a B.Tech final/mini project. It predicts stock prices using advanced Deep Learning architectures (LSTM & GRU), incorporates sentiment analysis from news headlines, and features a Streamlit desktop/web application for interactive predictions.

## Features
- **Data Pipeline**: Automated data fetching via `yfinance` including OHLCV data.
- **Feature Engineering**: Integration of Technical Indicators (SMA-20, SMA-50, RSI, MACD).
- **Deep Learning Models**: Sophisticated multi-layer LSTM and GRU models for time-series forecasting.
- **Ensemble Model**: Combines LSTM predictions with XGBoost and Sentiment Analysis for enhanced accuracy.
- **Strict Preprocessing**: Avoids data leakage by applying chronologically separated MinMax scaling.
- **Interactive UI**: Streamlit-based web dashboard to visualize predictions, model comparisons, and technical indicators.
- **Full Academic Artifacts**: Contains generated comprehensive Reports, PPT content, and Viva QA preparations in the `docs/` folder.

## Installation

1. Clone or download this repository.
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training & Evaluation (CLI)
You can use the main application entry point to train and visualize models without the web app.

```bash
python main.py --mode train --ticker AAPL
python main.py --mode evaluate --ticker AAPL
```

### 2. Streamlit Web Application
To run the interactive web interface:

```bash
streamlit run app/streamlit_app.py
```

## Project Structure
- `config/` - Settings and Hyperparameters.
- `src/` - Core Python modules (`data_collector`, `feature_engineer`, `models`, `sentiment`, etc.)
- `outputs/` - Saved Models, Data Scalers, and Generated Plots.
- `app/` - Streamlit User Interface.
- `docs/` - Academic Documentation (Report, PPT, Viva Questions).

## Disclaimer
*This project is built for educational (academic) purposes. The models and predictions should not be used for actual financial trading without rigorous due diligence.*
