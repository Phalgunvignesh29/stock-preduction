import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Config
from src.data_collector import StockDataCollector
from src.feature_engineer import FeatureEngineer
from src.preprocessor import DataPreprocessor
from src.models.lstm_model import LSTMPredictor
from src.models.gru_model import GRUPredictor
from src.sentiment.sentiment_analyzer import NewsSentimentAnalyzer

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Market AI", page_icon="📈", layout="wide")

# ─── Premium Dark Theme CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.stApp { background: #0d1117; color: #c9d1d9; }
[data-testid="stSidebar"] { background: #161b22 !important; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
.main-header { font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(90deg, #00d4ff, #7b2ff7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.sub-header { color: #8b949e; font-size: 0.95rem; margin-top: -10px; margin-bottom: 20px; }
.signal-buy  { background: linear-gradient(135deg, #0d4f2e, #0a3d22); border: 1px solid #238636; border-radius: 16px; padding: 28px; text-align: center; }
.signal-sell { background: linear-gradient(135deg, #4f0d0d, #3d0a0a); border: 1px solid #da3633; border-radius: 16px; padding: 28px; text-align: center; }
.signal-hold { background: linear-gradient(135deg, #4a3800, #3d2d00); border: 1px solid #e3b341; border-radius: 16px; padding: 28px; text-align: center; }
.metric-card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin-bottom: 8px; }
.chip { display: inline-block; background: rgba(0,212,255,0.1); border: 1px solid rgba(0,212,255,0.3);
    border-radius: 20px; padding: 4px 14px; font-size: 0.8rem; color: #00d4ff; margin: 2px; }
div[data-baseweb="tab-list"] { background: #161b22 !important; border-radius: 10px; padding: 4px; }
button[data-baseweb="tab"] { background: transparent !important; color: #8b949e !important; border-radius: 8px; }
button[data-baseweb="tab"][aria-selected="true"] { background: rgba(0,212,255,0.15) !important; color: #00d4ff !important; }
.stButton > button { background: linear-gradient(135deg, #00d4ff20, #7b2ff720) !important;
    color: #00d4ff !important; border: 1px solid #00d4ff60 !important;
    border-radius: 8px; font-weight: 600; width: 100%; }
.stButton > button:hover { background: linear-gradient(135deg, #00d4ff40, #7b2ff740) !important; }
[data-testid="metric-container"] { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 16px; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
SUGGESTED_STOCKS = {
    "🇮🇳 Indian": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "SBIN.NS", "HDFCBANK.NS", "WIPRO.NS", "BAJFINANCE.NS", "ADANIENT.NS"],
    "🇺🇸 US":     ["AAPL", "NVDA", "MSFT", "TSLA", "GOOGL", "META", "AMZN"],
}

# ─── Session State ────────────────────────────────────────────────────────────
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

# ─── Helper Functions ─────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_data(ticker):
    collector = StockDataCollector()
    df = collector.fetch_stock(ticker)
    if df is not None:
        resolved = df.attrs.get('resolved_ticker', ticker)
        engineer = FeatureEngineer()
        df = engineer.add_all_features(df)
        sentiment = NewsSentimentAnalyzer()
        df = sentiment.get_simulated_historical_sentiment(df)
        df.attrs['resolved_ticker'] = resolved
    return df

def generate_signals(df):
    """Generate BUY/SELL signals from RSI, MACD crossover, and SMA crossover."""
    df = df.copy()
    df['Signal'] = 0
    df['MACD_Sig'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # RSI extremes
    df.loc[df['RSI'] < 35, 'Signal'] = 1
    df.loc[df['RSI'] > 70, 'Signal'] = -1

    # MACD crossover
    cross_up = (df['MACD'] > df['MACD_Sig']) & (df['MACD'].shift(1) <= df['MACD_Sig'].shift(1))
    cross_dn = (df['MACD'] < df['MACD_Sig']) & (df['MACD'].shift(1) >= df['MACD_Sig'].shift(1))
    df.loc[cross_up, 'Signal'] = 1
    df.loc[cross_dn, 'Signal'] = -1

    # SMA Golden/Death Cross (strongest signal — overrides)
    golden = (df['SMA_20'] > df['SMA_50']) & (df['SMA_20'].shift(1) <= df['SMA_50'].shift(1))
    death  = (df['SMA_20'] < df['SMA_50']) & (df['SMA_20'].shift(1) >= df['SMA_50'].shift(1))
    df.loc[golden, 'Signal'] = 1
    df.loc[death,  'Signal'] = -1

    return df

def get_recommendation(df):
    """Score the stock across 4 indicators and return BUY/SELL/HOLD."""
    score = 0
    reasons = []
    l = df.iloc[-1]

    if l['RSI'] < 35:
        score += 2; reasons.append(f"✅ RSI **{l['RSI']:.1f}** — oversold, potential bounce")
    elif l['RSI'] > 70:
        score -= 2; reasons.append(f"🔴 RSI **{l['RSI']:.1f}** — overbought, potential pullback")
    else:
        reasons.append(f"⚪ RSI **{l['RSI']:.1f}** — neutral zone")

    if l['SMA_20'] > l['SMA_50']:
        score += 1; reasons.append("✅ SMA-20 **above** SMA-50 — bullish trend")
    else:
        score -= 1; reasons.append("🔴 SMA-20 **below** SMA-50 — bearish trend")

    if l['MACD'] > 0:
        score += 1; reasons.append("✅ MACD **positive** — upward momentum")
    else:
        score -= 1; reasons.append("🔴 MACD **negative** — downward momentum")

    if l['Close'] > l['SMA_20']:
        score += 1; reasons.append("✅ Price **above** 20-day average")
    else:
        score -= 1; reasons.append("🔴 Price **below** 20-day average")

    if score >= 3:    action, css = "🚀 STRONG BUY", "signal-buy"
    elif score >= 1:  action, css = "📈 BUY",        "signal-buy"
    elif score <= -3: action, css = "🚨 STRONG SELL", "signal-sell"
    elif score <= -1: action, css = "📉 SELL",        "signal-sell"
    else:             action, css = "⏸ HOLD",         "signal-hold"

    return action, score, reasons, css

@st.cache_data(ttl=120)
def quick_signal(ticker):
    """Lightweight ticker analysis for the suggestions panel."""
    try:
        collector = StockDataCollector()
        df = collector.fetch_stock(ticker, start_date="2023-01-01")
        if df is None or len(df) < 70:
            return None
        engineer = FeatureEngineer()
        df = engineer.add_all_features(df)
        action, score, _, _ = get_recommendation(df)
        resolved = df.attrs.get('resolved_ticker', ticker)
        display_name = ticker.replace('.NS', '').replace('.BO', '')
        return {
            "ticker": resolved,
            "display": display_name,
            "price":  df['Close'].iloc[-1],
            "change": df['Close'].iloc[-1] - df['Close'].iloc[-2],
            "rsi":    df['RSI'].iloc[-1],
            "action": action,
            "score":  score,
        }
    except Exception:
        return None

def generate_predictions(df, ticker, model_choice):
    resolved_ticker = df.attrs.get('resolved_ticker', ticker)
    preprocessor = DataPreprocessor()
    train_df, test_df = preprocessor.chronological_split(df)
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if not preprocessor.load_scaler(resolved_ticker):
        return None, None, None, f"No trained model for **{resolved_ticker}**. Run `python main.py --mode train --ticker {resolved_ticker}` first."
    train_df = train_df[num_cols]; test_df = test_df[num_cols]
    cols = list(train_df.columns)
    if 'Close' in cols:
        cols.remove('Close'); cols = ['Close'] + cols
    train_df = train_df[cols]; test_df = test_df[cols]
    preprocessor.target_col_idx = 0
    scaled_test = preprocessor.scaler.transform(test_df)
    X_test, y_test = preprocessor.create_sequences(scaled_test)
    input_shape = (X_test.shape[1], X_test.shape[2])
    model_path = os.path.join(Config.MODELS_DIR, f"{resolved_ticker}_{model_choice.lower()}_best.keras")
    if not os.path.exists(model_path):
        return None, None, None, f"No trained model at `{model_path}`. Please train first."
    model = LSTMPredictor.build_model(input_shape) if model_choice == "LSTM" else GRUPredictor.build_model(input_shape)
    model.load_weights(model_path)
    preds_scaled = model.predict(X_test)
    preds   = preprocessor.inverse_transform_target(preds_scaled)
    y_actual = preprocessor.inverse_transform_target(y_test.reshape(-1, 1))
    test_dates = test_df.index[Config.LOOKBACK_WINDOW:]
    return test_dates, y_actual, preds, None

def predict_future(df, ticker, model_choice, forecast_days=30):
    """
    Iterative multi-step forecast: uses the trained model recursively to predict
    the next `forecast_days` of prices beyond the last known data point.
    """
    resolved_ticker = df.attrs.get('resolved_ticker', ticker)
    preprocessor = DataPreprocessor()

    if not preprocessor.load_scaler(resolved_ticker):
        return None, None, f"No trained model for **{resolved_ticker}**. Train first."

    model_path = os.path.join(Config.MODELS_DIR, f"{resolved_ticker}_{model_choice.lower()}_best.keras")
    if not os.path.exists(model_path):
        return None, None, f"No trained model at `{model_path}`. Train first."

    # Prepare feature columns (same order as training: Close first)
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df_num = df[num_cols].copy()
    cols = list(df_num.columns)
    if 'Close' in cols:
        cols.remove('Close'); cols = ['Close'] + cols
    df_num = df_num[cols]
    preprocessor.target_col_idx = 0

    scaled_data = preprocessor.scaler.transform(df_num)
    lookback = Config.LOOKBACK_WINDOW
    input_seq = scaled_data[-lookback:].copy()  # (lookback, n_features)

    # Build & load model
    input_shape = (lookback, scaled_data.shape[1])
    model = LSTMPredictor.build_model(input_shape) if model_choice == "LSTM" else GRUPredictor.build_model(input_shape)
    model.load_weights(model_path)

    # Iterative prediction loop
    raw_preds = []
    current = input_seq.copy()
    for _ in range(forecast_days):
        X = current.reshape(1, lookback, scaled_data.shape[1])
        pred_scaled = model.predict(X, verbose=0)[0, 0]
        raw_preds.append(pred_scaled)
        # Slide window: drop oldest row, append new row with updated Close
        new_row = current[-1].copy()
        new_row[0] = pred_scaled
        current = np.vstack([current[1:], new_row])

    # Inverse-transform back to price scale
    preds_original = preprocessor.inverse_transform_target(np.array(raw_preds).reshape(-1, 1))

    # Generate business-day future dates
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    return future_dates, preds_original, None

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    ticker_input = st.text_input("Stock Ticker", value="AAPL").upper()
    st.caption("💡 Indian stocks: type RELIANCE, TCS, INFY — suffix auto-added.")
    model_choice = st.selectbox("Model Architecture", ("LSTM", "GRU"))
    run_prediction = st.button("🤖 Run Prediction Pipeline")

    st.markdown("---")
    st.markdown("### ⚡ Live Auto-Refresh")
    live_mode = st.toggle("Auto-refresh every 30s")
    if live_mode:
        # Non-blocking JS-based refresh — UI stays fully interactive
        st_autorefresh(interval=30_000, limit=None, key="stock_autorefresh")
        st.cache_data.clear()
        st.success(f"🟢 Live · Updated {datetime.now().strftime('%H:%M:%S')}")
    else:
        st.caption("Enable to fetch fresh data every 30s.")

    st.markdown("---")
    st.markdown("### 💡 Quick-Pick Tickers")
    for group, stocks in SUGGESTED_STOCKS.items():
        st.markdown(f"**{group}**")
        chips = " ".join([f'<span class="chip">{s.replace(".NS", "").replace(".BO", "")}</span>' for s in stocks])
        st.markdown(chips, unsafe_allow_html=True)

# ─── Main Header ──────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">📈 Stock Market AI</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="sub-header">Deep Learning Predictions · Buy/Sell Signals · Real-time Data'
    f'{" &nbsp;·&nbsp; 🟢 Live Mode · " + datetime.now().strftime("%H:%M:%S") if live_mode else ""}</p>',
    unsafe_allow_html=True
)

# ─── Data Load ────────────────────────────────────────────────────────────────
if not ticker_input:
    st.info("Enter a stock ticker in the sidebar to get started.")
    st.stop()

with st.spinner(f"Fetching data for {ticker_input}..."):
    df = load_data(ticker_input)

if df is None:
    st.error(f"❌ Failed to fetch data for **{ticker_input}**.\n\n"
             "For Indian stocks try: RELIANCE, TCS, INFY, SBIN, HDFCBANK, WIPRO")
    st.stop()

resolved = df.attrs.get('resolved_ticker', ticker_input)
if resolved != ticker_input:
    st.info(f"📍 Resolved **{ticker_input}** → **{resolved}** (Indian NSE auto-detected)")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview & Signal", "📈 Chart with Buy/Sell", "🤖 AI Prediction", "📡 Live Forecast", "💡 Stock Suggestions"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW & TRADING SIGNAL
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    latest = df['Close'].iloc[-1]
    prev   = df['Close'].iloc[-2]
    change = latest - prev
    chg_pct = (change / prev) * 100
    chg_col = "#3fb950" if change >= 0 else "#f85149"
    chg_arr = "▲" if change >= 0 else "▼"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div style="color:#8b949e;font-size:.78rem">Latest Close</div>
            <div style="font-size:1.85rem;font-weight:800">${latest:,.2f}</div>
            <div style="color:{chg_col};font-size:.88rem">{chg_arr} {abs(change):.2f} ({chg_pct:+.2f}%)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        sma20 = df['SMA_20'].iloc[-1]
        st.markdown(f"""<div class="metric-card">
            <div style="color:#8b949e;font-size:.78rem">SMA-20</div>
            <div style="font-size:1.85rem;font-weight:800">${sma20:,.2f}</div>
            <div style="color:#8b949e;font-size:.88rem">20-Day Average</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        rsi = df['RSI'].iloc[-1]
        rc = "#3fb950" if rsi < 35 else "#f85149" if rsi > 70 else "#e3b341"
        rl = "Oversold 🟢" if rsi < 35 else "Overbought 🔴" if rsi > 70 else "Neutral ⚪"
        st.markdown(f"""<div class="metric-card">
            <div style="color:#8b949e;font-size:.78rem">RSI (14)</div>
            <div style="font-size:1.85rem;font-weight:800;color:{rc}">{rsi:.1f}</div>
            <div style="color:{rc};font-size:.88rem">{rl}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        macd = df['MACD'].iloc[-1]
        mc = "#3fb950" if macd > 0 else "#f85149"
        ml = "Bullish Momentum 📈" if macd > 0 else "Bearish Momentum 📉"
        st.markdown(f"""<div class="metric-card">
            <div style="color:#8b949e;font-size:.78rem">MACD</div>
            <div style="font-size:1.85rem;font-weight:800;color:{mc}">{macd:+.3f}</div>
            <div style="color:{mc};font-size:.88rem">{ml}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 Current Trading Recommendation")

    action, score, reasons, css = get_recommendation(df)
    is_buy  = "BUY"  in action
    is_sell = "SELL" in action
    sig_color = "#3fb950" if is_buy else "#f85149" if is_sell else "#e3b341"
    sig_icon  = "🚀" if is_buy else "🚨" if is_sell else "⏸"

    sig_col, reasons_col = st.columns([1, 2])
    with sig_col:
        st.markdown(f"""<div class="{css}">
            <div style="font-size:3.5rem">{sig_icon}</div>
            <div style="font-size:1.7rem;font-weight:800;color:{sig_color};margin:8px 0">{action}</div>
            <div style="color:#8b949e;font-size:.85rem">Signal Score: {score:+d} / 4</div>
            <div style="color:#8b949e;font-size:.78rem;margin-top:8px">{resolved}</div>
        </div>""", unsafe_allow_html=True)
    with reasons_col:
        st.markdown("**Analysis Breakdown:**")
        for r in reasons:
            st.markdown(r)
        sentiment_val = df['Sentiment'].iloc[-1] if 'Sentiment' in df.columns else 0.0
        sent_label = "Positive 🟢" if sentiment_val > 0.05 else "Negative 🔴" if sentiment_val < -0.05 else "Neutral ⚪"
        st.markdown(f"📰 **News Sentiment**: `{sentiment_val:.3f}` — {sent_label}")

    # What this means for investor
    st.markdown("---")
    if is_buy:
        st.success(f"💰 **Action**: Consider **buying** {resolved}. Multiple indicators point to upward momentum. Set a stop-loss ~3-5% below entry price.")
    elif is_sell:
        st.error(f"⚠️ **Action**: Consider **selling** or reducing position in {resolved}. Indicators suggest bearish pressure.")
    else:
        st.warning(f"⏸ **Action**: **Hold** your current position in {resolved}. Mixed signals — wait for a clearer trend.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHART WITH BUY/SELL SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    df_sig = generate_signals(df)
    buy_pts  = df_sig[df_sig['Signal'] ==  1]
    sell_pts = df_sig[df_sig['Signal'] == -1]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        subplot_titles=(f'{resolved} — Price, SMAs & Signals', 'RSI (14)', 'MACD'),
        row_heights=[0.55, 0.2, 0.25]
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='Price',
        increasing_line_color='#3fb950', decreasing_line_color='#f85149',
        increasing_fillcolor='#3fb950', decreasing_fillcolor='#f85149'
    ), row=1, col=1)

    # SMAs
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='#e3b341', width=1.5), name='SMA-20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='#58a6ff', width=1.5), name='SMA-50'), row=1, col=1)

    # BUY signals — green upward triangles BELOW price bar
    if not buy_pts.empty:
        fig.add_trace(go.Scatter(
            x=buy_pts.index,
            y=buy_pts['Close'] * 0.975,
            mode='markers+text',
            name='BUY Signal',
            text=["B"] * len(buy_pts),
            textposition="bottom center",
            textfont=dict(size=8, color='#3fb950'),
            marker=dict(symbol='triangle-up', size=14, color='#3fb950',
                        line=dict(color='white', width=1))
        ), row=1, col=1)

    # SELL signals — red downward triangles ABOVE price bar
    if not sell_pts.empty:
        fig.add_trace(go.Scatter(
            x=sell_pts.index,
            y=sell_pts['Close'] * 1.025,
            mode='markers+text',
            name='SELL Signal',
            text=["S"] * len(sell_pts),
            textposition="top center",
            textfont=dict(size=8, color='#f85149'),
            marker=dict(symbol='triangle-down', size=14, color='#f85149',
                        line=dict(color='white', width=1))
        ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#bc8cff', width=1.5), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='#f85149', opacity=0.6, annotation_text="Overbought 70", row=2, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='#3fb950', opacity=0.6, annotation_text="Oversold 30",  row=2, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.02)", line_width=0, row=2, col=1)

    # MACD
    macd_colors = ['#3fb950' if v >= 0 else '#f85149' for v in df['MACD_Histogram']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Histogram'], marker_color=macd_colors, name='Histogram', opacity=0.6), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#00d4ff', width=1.5), name='MACD'), row=3, col=1)
    macd_signal_line = df['MACD'].ewm(span=9, adjust=False).mean()
    fig.add_trace(go.Scatter(x=df.index, y=macd_signal_line, line=dict(color='#f97316', width=1.2, dash='dot'), name='Signal Line'), row=3, col=1)

    fig.update_layout(
        height=720,
        paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
        font=dict(color='#c9d1d9', family='Inter'),
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor='rgba(13,17,23,0.8)', bordercolor='#30363d', borderwidth=1, font=dict(size=11)),
        margin=dict(t=50, b=10, l=10, r=10),
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor='#21262d', row=i, col=1, showgrid=True, zeroline=False)
        fig.update_yaxes(gridcolor='#21262d', row=i, col=1, showgrid=True, zeroline=False)

    st.plotly_chart(fig, width='stretch')

    lc, mc2, rc = st.columns(3)
    lc.markdown(f"🟢 **BUY signals:** {len(buy_pts)} detected")
    mc2.markdown(f"🔴 **SELL signals:** {len(sell_pts)} detected")
    rc.markdown(f"📅 **Data:** {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")

    st.info("📖 **How to read signals:** Green ▲ triangles = BUY opportunities (oversold RSI / MACD crossup / Golden cross). Red ▼ triangles = SELL signals (overbought RSI / MACD crossdown / Death cross).")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — AI PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"### 🤖 {model_choice} Model Evaluation on Test Data")
    st.markdown("Trained on 80% of historical data. Evaluated on held-out 20% test set.")

    if run_prediction:
        with st.spinner("Running model inference…"):
            dates, y_actual, y_preds, err = generate_predictions(df, ticker_input, model_choice)

        if err:
            st.warning(err)
            
            st.markdown("---")
            st.info("The AI needs to learn the patterns for this specific stock before it can predict. Click below to start training.")
            if st.button(f"⚙️ Train {model_choice} Model for {resolved} Now", key="_train_eval"):
                with st.spinner(f"Training {model_choice} model for {resolved}... This takes ~1 minute. Please wait..."):
                    import subprocess
                    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")
                    res = subprocess.run([sys.executable, script_path, "--mode", "train", "--ticker", resolved, "--model", model_choice], capture_output=True, text=True)
                    if res.returncode == 0:
                        st.success("✅ Training complete! Please click 'Run Prediction Pipeline' again.")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(f"❌ Training failed:\n\n{res.stderr}")
                        
        elif dates is not None:
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            rmse = np.sqrt(mean_squared_error(y_actual, y_preds))
            mae  = mean_absolute_error(y_actual, y_preds)
            mape = np.mean(np.abs((y_actual - y_preds) / (y_actual + 1e-8))) * 100

            m1, m2, m3 = st.columns(3)
            m1.metric("📉 RMSE", f"{rmse:.2f}", help="Root Mean Squared Error — avg deviation in price units")
            m2.metric("📊 MAE",  f"{mae:.2f}",  help="Mean Absolute Error")
            m3.metric("📈 MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error")

            pred_fig = go.Figure()
            pred_fig.add_trace(go.Scatter(
                x=dates, y=y_actual, mode='lines', name='Actual Price',
                line=dict(color='#58a6ff', width=2)
            ))
            pred_fig.add_trace(go.Scatter(
                x=dates, y=y_preds, mode='lines', name='Predicted Price',
                line=dict(color='#f97316', width=2, dash='dot')
            ))
            # Shade the area between them
            pred_fig.add_trace(go.Scatter(
                x=list(dates) + list(dates[::-1]),
                y=list(y_actual) + list(y_preds[::-1]),
                fill='toself', fillcolor='rgba(88,166,255,0.07)',
                line=dict(color='rgba(0,0,0,0)'), showlegend=False, name='Error Band'
            ))
            pred_fig.update_layout(
                title=f"Actual vs Predicted Stock Price — {model_choice}",
                paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
                font=dict(color='#c9d1d9', family='Inter'),
                xaxis=dict(gridcolor='#21262d'),
                yaxis=dict(gridcolor='#21262d', title='Price (USD)'),
                legend=dict(bgcolor='rgba(13,17,23,0.8)', bordercolor='#30363d', borderwidth=1),
            )
            st.plotly_chart(pred_fig, width='stretch')
            st.success(f"✅ Inference complete  |  RMSE: **{rmse:.2f}**  |  MAE: **{mae:.2f}**  |  MAPE: **{mape:.2f}%**")
    else:
        st.info("👈 Click **🤖 Run Prediction Pipeline** in the sidebar to generate AI predictions.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LIVE FORECAST (AI future prediction)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📡 Live AI Forecast — Next 30 Days")
    st.markdown(
        f"Using the **{model_choice}** model trained on **{resolved}** to predict the next 30 business days "
        f"beyond **{df.index[-1].strftime('%d %b %Y')}**."
    )

    if live_mode:
        st.markdown(f'<span style="color:#3fb950;font-size:.85rem">🟢 Auto-refreshing · {datetime.now().strftime("%H:%M:%S")}</span>', unsafe_allow_html=True)

    with st.spinner("Running AI forecast…"):
        future_dates, future_preds, forecast_err = predict_future(df, ticker_input, model_choice, forecast_days=30)

    if forecast_err:
        st.warning(forecast_err)
        
        st.markdown("---")
        st.info("The AI needs to be trained on this stock's historical data first so it can predict the future.")
        if st.button(f"⚙️ Train {model_choice} Model for {resolved} Now", use_container_width=True, key="_train_fcst"):
            with st.spinner(f"Training {model_choice} model for {resolved}... This takes ~1 minute. Please wait..."):
                import subprocess
                script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")
                res = subprocess.run([sys.executable, script_path, "--mode", "train", "--ticker", resolved, "--model", model_choice], capture_output=True, text=True)
                if res.returncode == 0:
                    st.success("✅ Training complete! Forecasting future prices...")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"❌ Training failed:\n\n{res.stderr}")
    else:
        current_price = df['Close'].iloc[-1]
        pred_7d  = future_preds[6]
        pred_14d = future_preds[13]
        pred_30d = future_preds[29]

        def pct(a, b): return ((b - a) / a) * 100
        def color(v): return "#3fb950" if v >= 0 else "#f85149"
        def arr(v):   return "▲" if v >= 0 else "▼"

        # Key milestone cards
        st.markdown("#### 🎯 Price Targets")
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            st.markdown(f"""<div class="metric-card" style="text-align:center">
                <div style="color:#8b949e;font-size:.78rem">Current Price</div>
                <div style="font-size:1.6rem;font-weight:800">${current_price:,.2f}</div>
                <div style="color:#8b949e;font-size:.82rem">Today</div>
            </div>""", unsafe_allow_html=True)
        with fc2:
            d = pct(current_price, pred_7d)
            st.markdown(f"""<div class="metric-card" style="text-align:center;border-color:{color(d)}40">
                <div style="color:#8b949e;font-size:.78rem">7-Day Target</div>
                <div style="font-size:1.6rem;font-weight:800">${pred_7d:,.2f}</div>
                <div style="color:{color(d)};font-size:.82rem">{arr(d)} {abs(d):.2f}%</div>
            </div>""", unsafe_allow_html=True)
        with fc3:
            d = pct(current_price, pred_14d)
            st.markdown(f"""<div class="metric-card" style="text-align:center;border-color:{color(d)}40">
                <div style="color:#8b949e;font-size:.78rem">14-Day Target</div>
                <div style="font-size:1.6rem;font-weight:800">${pred_14d:,.2f}</div>
                <div style="color:{color(d)};font-size:.82rem">{arr(d)} {abs(d):.2f}%</div>
            </div>""", unsafe_allow_html=True)
        with fc4:
            d = pct(current_price, pred_30d)
            st.markdown(f"""<div class="metric-card" style="text-align:center;border-color:{color(d)}40">
                <div style="color:#8b949e;font-size:.78rem">30-Day Target</div>
                <div style="font-size:1.6rem;font-weight:800">${pred_30d:,.2f}</div>
                <div style="color:{color(d)};font-size:.82rem">{arr(d)} {abs(d):.2f}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Build the combined chart — last 90 days historical + 30-day forecast
        hist_window = df.tail(90)
        hist_dates  = hist_window.index
        hist_prices = hist_window['Close'].values

        # Compute confidence bands (±std of last 30 days as proxy)
        recent_std = hist_prices[-30:].std()
        upper_band = future_preds + recent_std * 1.5
        lower_band = future_preds - recent_std * 1.5
        # Prevent negative prices
        lower_band = np.maximum(lower_band, 0)

        fig = go.Figure()

        # Historical prices
        fig.add_trace(go.Scatter(
            x=hist_dates, y=hist_prices,
            mode='lines', name='Historical Price',
            line=dict(color='#58a6ff', width=2.5)
        ))

        # Confidence band (shaded area)
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(upper_band) + list(lower_band[::-1]),
            fill='toself',
            fillcolor='rgba(249,115,22,0.12)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=True, name='Confidence Band'
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=future_dates, y=future_preds,
            mode='lines+markers', name=f'{model_choice} Forecast',
            line=dict(color='#f97316', width=2.5, dash='dot'),
            marker=dict(size=5, color='#f97316')
        ))

        # Connector from last historical point to first forecast point
        fig.add_trace(go.Scatter(
            x=[hist_dates[-1], future_dates[0]],
            y=[hist_prices[-1], future_preds[0]],
            mode='lines', name='',
            line=dict(color='#f97316', width=1.5, dash='dot'),
            showlegend=False
        ))

        # Vertical "Today" line
        today_dt = df.index[-1]
        fig.add_vline(x=today_dt, line_dash='dash', line_color='#8b949e')
        fig.add_annotation(
            x=today_dt, y=0.95, yref='paper',
            text="Today ", showarrow=False,
            font=dict(color='#8b949e', size=11),
            xanchor='right', xshift=-5
        )

        # Mark 7, 14, 30-day milestones
        for i, (label, idx) in enumerate({"7d": 6, "14d": 13, "30d": 29}.items()):
            fig.add_annotation(
                x=future_dates[idx], y=future_preds[idx],
                text=f"  {label}: ${future_preds[idx]:,.0f}",
                showarrow=True, arrowhead=2, arrowcolor='#f97316',
                font=dict(color='#f97316', size=11),
                bgcolor='rgba(13,17,23,0.8)', bordercolor='#f97316', borderwidth=1
            )

        fig.update_layout(
            title=f"{resolved} — AI Price Forecast (Next 30 Business Days)",
            height=520,
            paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
            font=dict(color='#c9d1d9', family='Inter'),
            xaxis=dict(gridcolor='#21262d', title='Date'),
            yaxis=dict(gridcolor='#21262d', title='Price'),
            legend=dict(bgcolor='rgba(13,17,23,0.8)', bordercolor='#30363d', borderwidth=1),
            margin=dict(t=60, b=20),
            hovermode='x unified'
        )
        st.plotly_chart(fig, width='stretch')

        # Trend verdict
        d30 = pct(current_price, pred_30d)
        if d30 >= 5:
            st.success(f"🚀 **Bullish Forecast**: The {model_choice} model projects a **{d30:.1f}% rise** over 30 days. Target: **${pred_30d:,.2f}**")
        elif d30 <= -5:
            st.error(f"📉 **Bearish Forecast**: The {model_choice} model projects a **{abs(d30):.1f}% drop** over 30 days. Target: **${pred_30d:,.2f}**")
        else:
            st.warning(f"⏸ **Sideways Forecast**: The {model_choice} model projects a **{d30:+.1f}%** movement over 30 days.")

        st.caption("⚠️ Forecast is based on technical patterns learned from historical data and does not account for news, earnings, or macro events. For educational purposes only.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — STOCK SUGGESTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 💡 Live Stock Recommendations")

    st.markdown("Real-time technical analysis across popular stocks. Updated every time the page loads.")

    all_tickers = []
    for g, stocks in SUGGESTED_STOCKS.items():
        all_tickers.extend(stocks)

    results = []
    progress = st.progress(0, text="Analyzing stocks…")
    for idx, t in enumerate(all_tickers):
        info = quick_signal(t)
        if info:
            results.append(info)
        progress.progress((idx + 1) / len(all_tickers), text=f"Analyzing {t}…")
    progress.empty()

    # Sort: BUY first, then HOLD, then SELL
    def sort_key(r):
        if "BUY" in r['action']:   return (0, -r['score'])
        elif "HOLD" in r['action']: return (1, 0)
        else:                       return (2, r['score'])
    results.sort(key=sort_key)

    buy_results  = [r for r in results if "BUY"  in r['action']]
    hold_results = [r for r in results if "HOLD" in r['action']]
    sell_results = [r for r in results if "SELL" in r['action']]

    if buy_results:
        st.markdown(f"#### 🟢 Buy Opportunities ({len(buy_results)})")
        cols = st.columns(min(4, len(buy_results)))
        for i, r in enumerate(buy_results):
            chg = r['change']
            chg_c = "#3fb950" if chg >= 0 else "#f85149"
            with cols[i % len(cols)]:
                st.markdown(f"""<div style="background:#0d4f2e;border:1px solid #238636;border-radius:14px;
                    padding:18px;text-align:center;margin-bottom:12px;">
                    <div style="font-size:.8rem;color:#8b949e">{r['display']}</div>
                    <div style="font-weight:700;font-size:1rem;color:#c9d1d9">{r['ticker']}</div>
                    <div style="font-size:1.4rem;font-weight:800;margin:8px 0">${r['price']:,.2f}</div>
                    <div style="font-size:.82rem;color:{chg_c}">{'▲' if chg>=0 else '▼'} {abs(chg):.2f}</div>
                    <div style="font-size:.78rem;color:#8b949e;margin-top:4px">RSI: {r['rsi']:.1f}</div>
                    <div style="font-size:.9rem;font-weight:700;color:#3fb950;margin-top:8px">{r['action']}</div>
                </div>""", unsafe_allow_html=True)

    if hold_results:
        st.markdown(f"#### 🟡 Hold / Neutral ({len(hold_results)})")
        cols = st.columns(min(4, len(hold_results)))
        for i, r in enumerate(hold_results):
            chg = r['change']
            chg_c = "#3fb950" if chg >= 0 else "#f85149"
            with cols[i % len(cols)]:
                st.markdown(f"""<div style="background:#1c1c1c;border:1px solid #30363d;border-radius:14px;
                    padding:18px;text-align:center;margin-bottom:12px;">
                    <div style="font-size:.8rem;color:#8b949e">{r['display']}</div>
                    <div style="font-weight:700;font-size:1rem;color:#c9d1d9">{r['ticker']}</div>
                    <div style="font-size:1.4rem;font-weight:800;margin:8px 0">${r['price']:,.2f}</div>
                    <div style="font-size:.82rem;color:{chg_c}">{'▲' if chg>=0 else '▼'} {abs(chg):.2f}</div>
                    <div style="font-size:.78rem;color:#8b949e;margin-top:4px">RSI: {r['rsi']:.1f}</div>
                    <div style="font-size:.9rem;font-weight:700;color:#e3b341;margin-top:8px">{r['action']}</div>
                </div>""", unsafe_allow_html=True)

    if sell_results:
        st.markdown(f"#### 🔴 Sell / Avoid ({len(sell_results)})")
        cols = st.columns(min(4, len(sell_results)))
        for i, r in enumerate(sell_results):
            chg = r['change']
            chg_c = "#3fb950" if chg >= 0 else "#f85149"
            with cols[i % len(cols)]:
                st.markdown(f"""<div style="background:#4f0d0d;border:1px solid #da3633;border-radius:14px;
                    padding:18px;text-align:center;margin-bottom:12px;">
                    <div style="font-size:.8rem;color:#8b949e">{r['display']}</div>
                    <div style="font-weight:700;font-size:1rem;color:#c9d1d9">{r['ticker']}</div>
                    <div style="font-size:1.4rem;font-weight:800;margin:8px 0">${r['price']:,.2f}</div>
                    <div style="font-size:.82rem;color:{chg_c}">{'▲' if chg>=0 else '▼'} {abs(chg):.2f}</div>
                    <div style="font-size:.78rem;color:#8b949e;margin-top:4px">RSI: {r['rsi']:.1f}</div>
                    <div style="font-size:.9rem;font-weight:700;color:#f85149;margin-top:8px">{r['action']}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("⚠️ Disclaimer: These signals are generated using technical analysis for **educational purposes only** and do not constitute financial advice. Always do your own research before investing.")
