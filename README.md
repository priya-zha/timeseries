# 📈 Stock Market Forecasting using Time Series Analysis (LSTM)

A Streamlit web application that forecasts stock prices using an LSTM (Long Short-Term Memory) deep learning model. Users select a custom date range and the app predicts future Tesla (TSLA) stock closing prices using time series analysis.

---

## 📌 Description

This project combines deep learning with financial data analysis to forecast stock market prices. It uses a pre-trained LSTM model loaded from `model.h5` to predict future closing prices of Tesla stock based on historical data fetched via the `yfinance` API. The results are displayed interactively using Plotly charts alongside a forecast data table — all within a Streamlit web interface.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application UI |
| TensorFlow / Keras | LSTM model loading and inference |
| yfinance | Fetching historical stock data from Yahoo Finance |
| Plotly | Interactive stock price charts |
| Scikit-learn (MinMaxScaler) | Data normalization for LSTM input |
| Pandas | Data manipulation and date handling |
| NumPy | Array operations for model input preparation |

---

## 📁 Project Structure

```
├── date.py       # Main Streamlit app for stock forecasting
├── model.h5      # Pre-trained LSTM model for stock price prediction
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/priya-zha/timeseries.git
cd timeseries
```

### 2. Install Dependencies
```bash
pip install streamlit tensorflow yfinance plotly scikit-learn pandas numpy
```

### 3. Run the App
```bash
streamlit run date.py
```

---

## 🚀 Usage

1. Open the app in your browser (Streamlit will provide a local URL, typically `http://localhost:8501`)
2. Select a **From Date** and **To Date** using the date pickers
3. Click **Generate** to run the LSTM forecast
4. View the interactive Plotly chart showing:
   - Original Tesla stock price history
   - Forecasted future prices (in red)
5. A forecast data table is also displayed alongside the chart

---

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.x
- Streamlit
- yfinance
- Plotly
- Scikit-learn
- Pandas, NumPy, LSTM

---

## 👩‍💻 Author

**Priya** — [@priya-zha](https://github.com/priya-zha)
