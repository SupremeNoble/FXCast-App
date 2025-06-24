# ----------------------- Manual ADOSC Calculation -----------------------
def calculate_adosc(df):
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"])
    clv = clv.replace([np.inf, -np.inf], 0).fillna(0)
    ad_line = clv * df["Volume"]
    return ad_line.cumsum()

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import SMAIndicator, WMAIndicator, MACD, CCIIndicator
#from ta.volatility import AverageTrueRange

# ----------------------- Feature Calculation -----------------------
def calculate_features(df):
    df["SMA_14"] = SMAIndicator(df["Close"], window=14).sma_indicator()
    df["WMA_14"] = WMAIndicator(df["Close"], window=14).wma()
    df["Momentum_14"] = df["Close"].diff(1)  # renamed
    stoch = StochasticOscillator(df["High"], df["Low"], df["Close"], window=14)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()
    df["RSI_14"] = RSIIndicator(df["Close"], window=14).rsi()
    macd = MACD(df["Close"])
    df["MACD"] = macd.macd_diff()
    df["Williams_R"] = WilliamsRIndicator(df["High"], df["Low"], df["Close"], lbp=14).williams_r()  # renamed
    df["A/D_Osc"] = calculate_adosc(df)  # renamed
    df["CCI_14"] = CCIIndicator(df["High"], df["Low"], df["Close"], window=14).cci()  # renamed
    return df

# ----------------------- Discretization -----------------------
def discretize_latest(df):
    latest = df.iloc[-1:].copy()
    latest["SMA_14"] = 1 if latest["Close"].values[0] > latest["SMA_14"].values[0] else -1
    latest["WMA_14"] = 1 if latest["Close"].values[0] > latest["WMA_14"].values[0] else -1
    latest["Momentum_14"] = 1 if latest["Momentum_14"].values[0] > 0 else -1
    latest["Stoch_K"] = 1 if latest["Stoch_K"].values[0] > 50 else -1
    latest["Stoch_D"] = 1 if latest["Stoch_D"].values[0] > 50 else -1
    latest["RSI_14"] = 1 if latest["RSI_14"].values[0] > 50 else -1
    latest["MACD"] = 1 if latest["MACD"].values[0] > 0 else -1
    latest["Williams_R"] = 1 if latest["Williams_R"].values[0] > -50 else -1
    latest["A/D_Osc"] = 1 if latest["A/D_Osc"].values[0] > 0 else -1
    latest["CCI_14"] = 1 if latest["CCI_14"].values[0] > 0 else -1

    return latest[[
        "SMA_14", "WMA_14", "Momentum_14", "Stoch_K", "Stoch_D",
        "RSI_14", "MACD", "Williams_R", "A/D_Osc", "CCI_14"
    ]]

def discretize_bulk(df):
    disc = pd.DataFrame(index=df.index)
    disc["SMA_14"] = np.where(df["Close"] > df["SMA_14"], 1, -1)
    disc["WMA_14"] = np.where(df["Close"] > df["WMA_14"], 1, -1)
    disc["Momentum_14"] = np.where(df["Momentum_14"] > 0, 1, -1)
    disc["Stoch_K"] = np.where(df["Stoch_K"] > 50, 1, -1)
    disc["Stoch_D"] = np.where(df["Stoch_D"] > 50, 1, -1)
    disc["RSI_14"] = np.where(df["RSI_14"] > 50, 1, -1)
    disc["MACD"] = np.where(df["MACD"] > 0, 1, -1)
    disc["Williams_R"] = np.where(df["Williams_R"] > -50, 1, -1)
    disc["A/D_Osc"] = np.where(df["A/D_Osc"] > 0, 1, -1)
    disc["CCI_14"] = np.where(df["CCI_14"] > 0, 1, -1)
    return disc

# ----------------------- Load Models -----------------------
@st.cache_resource
def load_models():
    svm = joblib.load("Models/best_discrete_svm_model.pkl")
    rf = joblib.load("Models/best_discrete_rf_model.pkl")
    cb = joblib.load("Models/best_discrete_catboost_model.pkl")
    return svm, rf, cb

# Load the logo image
logo = Image.open("fxcast_logo.png")

# Center the logo using columns
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(logo, use_container_width=True)

df = pd.read_csv("yml.csv", sep="\t")
df.columns = ["Datetime", "Open", "High", "Low", "Close", "Volume", "Spread"]
df["Datetime"] = pd.to_datetime(df["Datetime"])

# Feature Engineering
df = calculate_features(df).dropna()

df["y"] = df["Close"].shift(1)
df["y"] = df.apply(
    lambda row: 1 if row["y"] > row["Close"] else (-1 if row["y"] < row["Close"] else np.nan),
    axis=1
)
df = df.dropna(subset=["y"]).reset_index(drop=True)
df["y"] = df["y"].astype(int)

features = discretize_latest(df)

# --- Chart Header ---
st.subheader("📉 AUD/USD Price Chart")

# --- Make sure Datetime column is in datetime format ---
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

# --- Filter and Prepare Data ---
df_temp = df.copy()
df_temp = df_temp.dropna(subset=["Datetime"])  # in case of parsing errors
df_temp.set_index("Datetime", inplace=True)
df_temp.sort_index(inplace=True)

last_date = df_temp.index.max()

# --- Latest Price and % Change Display ---
latest_close = df_temp["Close"].iloc[-1]
previous_close = df_temp["Close"].iloc[-2]

price_change = latest_close - previous_close
percent_change = (price_change / previous_close) * 100

# Color the text green for gain, red for loss
if price_change > 0:
    st.markdown(
        f"""<div style='font-size: 24px; font-weight: bold;'>
                Latest Close: ${latest_close:.4f} <span style='color: limegreen;'>+{price_change:.4f} (+{percent_change:.2f}%)</span>
            </div>""",
        unsafe_allow_html=True
    )
elif price_change < 0:
    st.markdown(
        f"""<div style='font-size: 24px; font-weight: bold;'>
                Latest Close: ${latest_close:.4f} <span style='color: red;'>{price_change:.4f} ({percent_change:.2f}%)</span>
            </div>""",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f"""<div style='font-size: 24px; font-weight: bold;'>
                Latest Close: ${latest_close:.4f} <span style='color: gray;'>No change</span>
            </div>""",
        unsafe_allow_html=True
    )

# --- Select Period (Yahoo-style) ---
time_window = st.selectbox("Select Period", ["1M", "6M", "1Y"], index=0)

# Define just the time range and line width (no static colors)
if time_window == "1M":
    start_date = last_date - pd.DateOffset(months=1)
    line_width = 1.5
elif time_window == "6M":
    start_date = last_date - pd.DateOffset(months=6)
    line_width = 2.0
elif time_window == "1Y":
    start_date = last_date - pd.DateOffset(years=1)
    line_width = 2.5

# Filter data
df_filtered = df_temp.loc[start_date:]

# Auto-zoom y-axis range around min/max prices with a small buffer
y_min = df_filtered["Close"].min()
y_max = df_filtered["Close"].max()
y_buffer = (y_max - y_min) * 0.05  # 5% margin

y_range = [y_min - y_buffer, y_max + y_buffer]

# Determine overall trend to color the entire line
start_price = df_filtered["Close"].iloc[0]
end_price = df_filtered["Close"].iloc[-1]

if end_price > start_price:
    line_color = "green"
    fill_color = "rgba(0,255,0,0.2)"
else:
    line_color = "red"
    fill_color = "rgba(255,0,0,0.2)"

# Create figure
fig = go.Figure()

# Create chart with one consistent color
fig.add_trace(go.Scatter(
    x=df_filtered.index,
    y=df_filtered["Close"],
    mode="lines",
    name="Close Price",
    line=dict(color=line_color, width=line_width),
    fill="tozeroy",
    fillcolor=fill_color
))

# Update layout
fig.update_layout(
    title=f"{time_window} AUD/USD Daily Closing Prices",
    xaxis_title="Date",
    yaxis_title="Close Price (USD)",
    yaxis=dict(range=y_range),
    template="plotly_dark",
    height=400
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("📈 Latest Discretized Indicators")

# Prepare formatted display of latest discretized indicators
# Transpose and format the DataFrame
display_df = features.T.reset_index()
display_df.columns = ["Technical Indicators", "Value"]

# Display in Streamlit
st.dataframe(display_df, use_container_width=True)

# Load models and predict
svm_model, rf_model, cb_model = load_models()
X = features.values

prediction_map = {-1: "⬇️ Down", 1: "⬆️ Up"}

# Extract date from the last row
latest_date = pd.to_datetime(df.iloc[-1]["Datetime"]) + pd.Timedelta(days=1)
st.subheader(f"📢 Prediction for {latest_date.date()}")

col1, col2, col3 = st.columns(3)
with col1:
    pred = svm_model.predict(X)[0]
    st.metric("SVM Prediction", prediction_map[pred])
with col2:
    pred = rf_model.predict(X)[0]
    st.metric("Random Forest", prediction_map[pred])
with col3:
    pred = cb_model.predict(X)[0]
    st.metric("CatBoost", prediction_map[pred])

# ----------------------- Project Accuracy Donut Charts -----------------------
st.subheader("📊 Model Accuracy")

# Accuracy values (as percentages)
accuracy_scores = {
    "SVM": 89.5,
    "Random Forest": 89.3,
    "CatBoost": 89.7
}

# Donut chart plotting function with accuracy in center
def plot_donut_chart(accuracy, label):
    incorrect = 100 - accuracy
    fig, ax = plt.subplots()
    wedges, texts = ax.pie(
        [accuracy, incorrect],
        labels=["", ""],
        colors=["#4CAF50", "#F44336"],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.4)
    )
    # Center label
    ax.text(0, 0, f"{accuracy:.1f}%", ha='center', va='center', fontsize=14, fontweight='bold')
    ax.set_title(label)
    return fig

# Display the charts side by side
col1, col2, col3 = st.columns(3)
with col1:
    fig = plot_donut_chart(accuracy_scores["SVM"], "SVM Accuracy")
    st.pyplot(fig)

with col2:
    fig = plot_donut_chart(accuracy_scores["Random Forest"], "Random Forest Accuracy")
    st.pyplot(fig)

with col3:
    fig = plot_donut_chart(accuracy_scores["CatBoost"], "CatBoost Accuracy")
    st.pyplot(fig)

# ----------------------- Actual vs Predicted Table -----------------------
st.subheader("🧾 Past Prediction Results")

# Prepare full dataset predictions
df_valid = df.dropna(subset=["y"]).copy()
X_all = discretize_bulk(df_valid)

# Predict using all rows with y
y_true_all = df_valid["y"].values
y_pred_svm_all = svm_model.predict(X_all)
y_pred_rf_all = rf_model.predict(X_all)
y_pred_cb_all = cb_model.predict(X_all)

# Map +1/-1 to string
label_map = {1: "⬆️ Up", -1: "⬇️ Down"}

result_table = pd.DataFrame({
    "Date": df_valid["Datetime"].dt.date,
    "Actual Movement": [label_map[y] for y in y_true_all],
    "SVM Prediction": [label_map[y] for y in y_pred_svm_all],
    "RF Prediction": [label_map[y] for y in y_pred_rf_all],
    "CatBoost Prediction": [label_map[y] for y in y_pred_cb_all],
})

# Show most recent first
result_table = result_table[::-1].reset_index(drop=True)
st.dataframe(result_table, use_container_width=True)
