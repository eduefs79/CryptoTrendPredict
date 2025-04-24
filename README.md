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

## 🤖 What's Next

- Animate Louvain clusters over time (rolling windows)
- Score and track BTC’s influence in the network via graph centrality
- Explore hybrid models (LSTM + Attention) and multistep forecasting
- Expose the best model as a REST API or Power BI dashboard

---

Made with ❤️ and insomnia.

