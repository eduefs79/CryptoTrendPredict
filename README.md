
# 📊 CryptoTrendPredict

**Predicting the next-day closing price of Bitcoin using technical indicators, sentiment indexes, and machine learning — with full integration into Snowflake and rich model diagnostics.**

---

## 🚀 Project Overview

CryptoTrendPredict builds a predictive pipeline to estimate Bitcoin's next-day closing price.  
It combines real-time data ingestion (from [CoinGecko](https://www.coingecko.com/) and [Alternative.me Fear & Greed Index](https://alternative.me/crypto/fear-and-greed-index/)), financial indicators, and machine learning models, while storing results in Snowflake for analysis and auditability.

---

## 🧠 Features

- ✅ Historical BTC-USD price ingestion via **CoinGecko API**
- ✅ Crypto **Fear & Greed Index integration**
- ✅ **Feature engineering** (MACD, RSI, MFI, Bollinger Bands, volatility, etc.)
- ✅ **Linear Regression modeling** with sklearn and statsmodels
- ✅ Full error diagnostics: `MAE`, `RMSE`, `R²`, and percent error
- ✅ **Plotly visualizations**: actual vs predicted, rolling errors, KDE plots, and percent errors
- ✅ Data persistence in **Snowflake**
- ✅ Markdown-based **report generation**
- ✅ Production-ready structure for GitHub and interviews

---

## 📸 Sample Visuals

| Actual vs Predicted | Rolling MAE & RMSE | % Error Over Time |
|---------------------|--------------------|-------------------|
| ![](assets/actual_vs_predicted.png) | ![](assets/rolling_errors.png) | ![](assets/error_pct.png) |

| RMSE KDE Distribution | MAE KDE Distribution |
|-----------------------|----------------------|
| ![](assets/kde_rmse.png) | ![](assets/kde_mae.png) |

---

## 📦 Requirements

- Python 3.10+
- [Docker](https://www.docker.com/get-started) (optional if using MySQL container)
- Snowflake account (for cloud-based DB storage)
- A `.env` file with your secrets (see `.env.example`)

---

## ⚙️ Setup Instructions

```bash
# Clone the repo
git clone https://github.com/eduefs79/CryptoTrendPredict.git
cd CryptoTrendPredict

# Install dependencies
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file with the following:

```ini
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ACCOUNT=your_account_id
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=CryptoDB
SNOWFLAKE_SCHEMA=PUBLIC
SNOWFLAKE_ROLE=ACCOUNTADMIN
PRIVATE_KEY_PATH=./keys/private_key.p8
```

---

## 🧪 How It Works

1. Downloads and merges BTC price + FGI sentiment data
2. Creates technical features and target column
3. Scales features, trains a Linear Regression model
4. Tracks performance using R², MAE, RMSE, MAE%
5. Saves all outputs to Snowflake
6. Visualizes model quality with rich charts
7. Exports a full Markdown report for audit

---

## 📊 Model Evaluation Example

```
R² Score      : 0.9953
MAE           : $1,079.24
RMSE          : $1,652.89
Avg MAE (%)   : 2.17%
Avg RMSE (%)  : 3.12%
```

---

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.

---

## 🙌 Credits

Built by [@eduefs79](https://github.com/eduefs79) with the help of ChatGPT as a co-pilot.
