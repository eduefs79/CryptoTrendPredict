# 📈 Bitcoin Price Prediction with Financial Indicators

This project uses multiple financial indicators, macroeconomic data (NASDAQ, S&P 500, VIX), and technical features (e.g., Bollinger Bands) to build and evaluate a machine learning model predicting the next-day price of Bitcoin.

## 🚀 Features
- Fetches and stores data in **Snowflake**
- Incorporates:
  - Bitcoin OHLCV data
  - Fear and Greed Index
  - NASDAQ, S&P 500, VIX
  - Bollinger Bands, Returns, Volatility
- Performs data cleaning and joins using SQL
- Linear Regression modeling with p-value feature selection
- Rich visualizations:
  - Actual vs. Predicted
  - MAE/RMSE % over time
  - Residuals, Error distribution
  - Feature correlation & significance

## 🛠️ Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## 📁 Folder Structure

```
.
├── Bitcoin.ipynb        # Main notebook with full pipeline
├── README.md            # Project overview
├── requirements.txt     # Python dependencies
└── assets/              # Exported charts and plots
```

## 🧠 Model Insights
- R² ≈ 0.995
- Average MAE ≈ $1,000
- Most impactful features: `Close`, `NASDAQ`, `Bollinger Bands`
- S&P 500 and VIX had low significance for short-term BTC prediction.

## 📊 Visual Output Samples
<img src="assets/actual_vs_predicted.png" width="600"/>
<img src="assets/error_percent_vs_time.png" width="600"/>
<img src="assets/feature_significance.png" width="600"/>

## 📌 Notes
- Ensure your Snowflake account has the correct roles & table permissions.
- FRED API used for macro indicators (S&P 500, VIX).
- VPNs may block `yfinance` and some endpoints.

---

Made with ❤️ and insomnia.
