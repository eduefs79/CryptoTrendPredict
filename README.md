# 🧠 Crypto Price Prediction & Network Analysis

This project applies technical analysis, machine learning, and network science to model and forecast Bitcoin price behavior using signals from its correlated crypto ecosystem and macroeconomic context.

---

## 🚀 Project Highlights

- ✅ BTC price prediction **Linear Regression** and **LSTM**
- 📊 Technical indicators computed using the `ta` library (MACD, RSI, MFI, EMA, etc.)
- 🌐 Integration with **Snowflake** for crypto data staging, historical persistence, and time series upserts
- 🧹 Cluster discovery using:
  - K-Means (based on return patterns)
  - Agglomerative (correlation distance)
  - **Louvain** (graph community detection on correlation networks)
- 🌐 Network graph visualization of crypto relationships and influence
- 🧮 Use of **macroeconomic indicators**, including:
  - Gold (GC=F), Silver (SI=F), S&P 500 (^GSPC), Nasdaq (^NDX), VIX
  - On-chain metrics: hash rate, miner revenue, active addresses, days destroyed
- 🔬 Feature selection using **p-values** from regression analysis
- 📈 Model evaluation via:
  - R² Score
  - RMSE
  - Actual vs Predicted line charts
  - Residual analysis

---

## 🔧 Features

- Fetches and stores data in **Snowflake**
- Uses cluster-based filtering for predictive features
- Combines traditional market, blockchain metrics, and crypto-specific TA

### 🏩 Macroeconomic + On-chain Data

- Traditional assets:
  - S&P 500 (^GSPC), Nasdaq (^NDX), Gold (GC=F), Silver (SI=F), VIX
- On-chain metrics from [blockchain.com API]:
  - Hash rate, miner revenue, Bitcoin days destroyed, difficulty, active addresses
- BTC-to-local-currency tickers:
  - BTC-INR, BTC-GBP, BTC-RUB, etc.

---

## 🛠️ Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📁 Folder Structure

```
.
├── crypto.ipynb         # Main notebook with full pipeline
├── utilities.py         # Utility functions for TA, clustering, Snowflake, etc.
├── Assets_Categorized.csv # Categorized asset metadata
├── README.md            # Project overview
├── requirements.txt     # Python dependencies
└── assets/              # Exported charts and plots
```

---

## 🔄 Dataflow Overview

The following diagram shows the overall pipeline from data ingestion to model evaluation:


```mermaid
flowchart TD
    A[Data Ingestion] --> B[Raw OHLCV]
    B --> C[Data Cleaning]
    C --> D1[Pivoted Prices]
    C --> D2[TA Features]
    D1 --> E[Louvain Clustering]
    D2 --> F[Linear Regression]
    E --> G[Peer Selection]
    G --> F
    G --> H[LSTM Model]
    F --> I[Model Metrics]
    H --> I
    I --> J[Evaluation Summary]
```

---

## 🧠 Model Insights

### ✅ Linear Regression (with reduced features)

- R² Score: **0.9756**
- RMSE: **\$2412**

### ✅ LSTM (price delta prediction)

- R² Score: **0.9826**
- RMSE: **\$2097**

### 💡 Observations

- Predicting **price delta** (instead of raw price) significantly improved LSTM's stability and interpretability.
- **LSTM now outperforms linear regression**, especially in volatile windows.
- Models trained with **Louvain-filtered features** show higher signal strength by reducing noise from low-correlation assets.
- All performance metrics are computed with **inverse-transformed prices** to preserve dollar-accuracy.

---

## 📌 Notes

- Ensure your Snowflake user has correct **roles and permissions**
- [FRED API] and `yfinance` are used for macro and market data
- VPNs or regional restrictions may block some endpoints

---

## 🔐 Snowflake Key Pair Authentication Setup

This project uses **key pair authentication** to securely connect to Snowflake.

### 🔧 Step 1: Generate Public/Private Key Pair

#### For Linux/macOS (OpenSSL):

```bash
openssl genrsa -out rsa_key.pem 2048
openssl rsa -in rsa_key.pem -pubout -out rsa_key.pub
```

#### For Windows (using PowerShell + OpenSSL):

1. Install [OpenSSL for Windows](https://slproweb.com/products/Win32OpenSSL.html)
2. Run:

```powershell
openssl genrsa -out rsa_key.pem 2048
openssl rsa -in rsa_key.pem -pubout -out rsa_key.pub
```

> 📝 Or use PuTTYgen (convert to PEM format)

---

### 🔐 Step 2: Upload Public Key to Snowflake

```sql
ALTER USER your_user_name SET RSA_PUBLIC_KEY='your_public_key_contents';
```

> ⚠️ Remove the `-----BEGIN PUBLIC KEY-----` and `END` lines

---

### ⚙️ Step 3: Add `.env` Variables

```ini
PRIVATE_KEY_PATH=/path/to/rsa_key.pem
SNOWFLAKE_ACCOUNT=your_account_id
SNOWFLAKE_USER=your_user_name
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=CryptoDB
SNOWFLAKE_SCHEMA=PUBLIC
SNOWFLAKE_ROLE=ACCOUNTADMIN
```

---

### ✅ You're Ready!

Your notebook will now securely connect to Snowflake using your key.

📚 [Snowflake Key Pair Auth Docs](https://docs.snowflake.com/en/user-guide/key-pair-auth)

---

## 🧠 Built With

- `pandas`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`
- `ta` & `talib` for technical analysis
- `networkx` & `python-louvain` for crypto correlation graphs
- `yfinance`, CoinGecko API, blockchain.com for data ingestion

---

## 🔬 Advanced Analyses and Enhancements

- 📹 Animate Louvain clusters over time using rolling windows
- 🧠 Score and monitor BTC’s influence using graph centrality metrics (eigenvector, betweenness)
- 🤖 Explore hybrid deep learning models (e.g., LSTM + Attention) and multi-step forecasting
- 📉 Evaluate multicollinearity using Variance Inflation Factor (VIF)
- 🧩 Model explainability with SHapley Additive exPlanations (SHAP)
  - 🧮 Applied SHAP to linear regression model to interpret feature importance. Bollinger Bands and EMAs were found to be the most influential predictors of BTC price.
- 📊 Benchmark performance against naive models (e.g., previous value, moving average)
- 📈 Visualize true vs predicted prices, residual trends, and percentage-based errors
- 🌐 Expose the best-performing model as a REST API or Power BI dashboard
- 📎 Expand SHAP explainability to include traditional assets (e.g., AAPL, MSFT, GLD) that may indirectly influence BTC, even if excluded from Louvain clusters.
- 🧠 Implement top-N token filtering or feature pruning to enable SHAP analysis without hitting GPU memory constraints.
- ☁️ Test high-dimensional SHAP explanations using Google Colab Pro (A100 GPU) or local distributed inference environments.


---

### 🔍 SHAP Summary: LSTM Feature Importance

The SHAP analysis reveals that XRP and ETH dominate short-term predictive influence on BTC price. 
XRP-related signals from `t-2` to `t-6` are especially prominent, indicating strong short-horizon correlation. ETH follows closely, also contributing meaningfully across several recent timesteps.

These results highlight how inter-token dynamics, especially among large-cap assets, drive short-term price movements in Bitcoin. The LSTM model is successfully capturing these temporal dependencies across the clustered peer network.

![SHAP Summary](shap_lstm_v1.png)

### 🌐 Crypto Influence Map

The influence map aggregates SHAP values by token and visualizes their direct influence on BTC prediction.

Louvain filtering was applied to isolate high-correlation assets, improving SHAP interpretability and reducing noise.

![Crypto Influence Map](shap_crypto_influence.png)

---
## 📉 Why Time Series Modeling Was Intentionally Excluded

While traditional time series models (e.g., ARIMA, Prophet) are commonly used in financial forecasting, I opted not to integrate them directly into this crypto prediction pipeline. Instead, the model leverages technical indicators, market network clustering (e.g., Louvain method), and regression techniques that already encapsulate key temporal dynamics, such as trend and volatility.

Time series models often assume a degree of stationarity or autocorrelation that is weak or unstable in high-volatility assets like cryptocurrencies. Given the already robust performance of the current feature set and the noisy nature of crypto markets, time series forecasting was evaluated but not adopted as it would likely contribute marginal additional signal. However, it remains a complementary option for long-term portfolio modeling or for generating probabilistic return bands in future iterations.

Made with ❤️ and insomnia.

### 🚧 Known Limitation

Due to the memory complexity of SHAP with full LSTM input (60 timesteps × 140+ tokens), 
explanations are currently computed only within the Louvain-filtered cluster around BTC. 
This improves model interpretability and reduces dimensionality for SHAP.



