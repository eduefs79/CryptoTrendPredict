# 🧠 Crypto Price Prediction & Network Analysis

This project applies technical analysis, machine learning, and network science to model and forecast Bitcoin price behavior using signals from its correlated crypto ecosystem and macroeconomic context.

---

## 🚀 Project Highlights

- ✅ BTC price prediction using **Random Forest** and **Linear Regression**
- 📊 Technical indicators computed using the `ta` library (MACD, RSI, MFI, EMA, etc.)
- 🌐 Integration with **Snowflake** for crypto data staging, historical persistence, and time series upserts
- 🧩 Cluster discovery using:
  - K-Means (based on return patterns)
  - Agglomerative (correlation distance)
  - **Louvain** (graph community detection)
- 🌐 Network graph visualization of crypto relationships and influence
- 🧮 Use of **macroeconomic indicators**, including:
  - Gold (GC=F), Silver (SI=F), S&P 500 (^GSPC), Nasdaq (^NDX), VIX
  - On-chain metrics: hash rate, miner revenue, active addresses, days destroyed
- 🔬 Feature selection using **p-values** from regression analysis
- 📈 Model evaluation via:
  - R² Score
  - RMSE
  - Confusion Matrix (with interpretation labels)
  - Actual vs Predicted line charts
  - Residual analysis

---

## 🔧 Features

- Fetches and stores data in **Snowflake**
- Uses cluster-based filtering for predictive features
- Combines traditional market, blockchain metrics, and crypto-specific TA

### 🏛️ Macroeconomic + On-chain Data

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

- R² ≈ **0.9713**
- RMSE ≈ **$2604**
- Feature reduction based on p-values improved both metrics

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

Made with ❤️ and insomnia.