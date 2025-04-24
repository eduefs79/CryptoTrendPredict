import os
import snowflake.connector
from cryptography.hazmat.primitives import serialization
from dotenv import load_dotenv
import yfinance as yf
import time
import datetime
import pandas as pd
import ta 
import talib
import snowflake.connector
from io import BytesIO
from tempfile import NamedTemporaryFile
from cryptography.hazmat.primitives import serialization
from sqlalchemy import text
from sqlalchemy.types import Float, DateTime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import community  # aka python-louvain


def upload_file(df: pd.DataFrame, stage_area: str, private_key,filename: str = 'default.parquet') -> None:
    """
    Uploads a DataFrame as a Parquet file to a Snowflake stage.

    Parameters:
        df (pd.DataFrame): DataFrame to upload
        stage_area (str): The Snowflake stage (e.g. @stage_name/path)
        private_key: The loaded private key object
    """

    # Save DataFrame to a temporary Parquet file
    tmp_path = filename
    df.to_parquet(tmp_path, index=False)

    # Convert private key to DER format (no encryption)
    private_key_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    db = os.getenv("SNOWFLAKE_DATABASE")
    # Connect to Snowflake
    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        private_key=private_key_bytes,
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role=os.getenv("SNOWFLAKE_ROLE")
    )

    print(f"✅ Connected to database: {db}")
    
    # Upload the file to the Snowflake stage
    with conn.cursor() as cur:
        put_sql = f"""
            PUT file://{tmp_path} {stage_area}
            AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
        """
        cur.execute(put_sql)
        print(f"✅ File uploaded to {stage_area}: {tmp_path}")

    # Clean up the temp file
    os.remove(tmp_path)
    print(f"🧹 Temp file removed: {tmp_path}")







def download_yahoo_to_stage(
    ticker: str,
    private_key,
    stage_area: str,
    interval: str = "1d",
    start: str = "2015-01-01",
    time: str = "12:00 AM",
    execution_time: str = "20250410172618"
) -> None:

    print(f"🔍 Downloading data for {ticker} from Yahoo Finance...")
    df = yf.download(ticker, interval=interval, start=start)

    # If Yahoo fails, try CoinMarketCap
    if df.empty:
        print(f"⚠️ No data returned for {ticker} from Yahoo Finance.")

        if '-' in ticker and ticker.split('-')[0] in ['BTC', 'ETH', 'LTC']:  # Add more cryptos if needed
            print(f"🔄 Trying CoinMarketCap for {ticker}...")
            df = get_coingecko_crypto_data(
                    ticker=ticker,
                    start=start,
                    end=pd.Timestamp.today().strftime('%Y-%m-%d'),
                    interval=interval
                )
        if df.empty:
            print(f"⛔ Still no data. Skipping {ticker}")
            return

    # ✅ Continue with formatting
    df.reset_index(inplace=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

    df.columns = [col.replace(f"_{ticker}", "") for col in df.columns]
    df.columns = [col.replace(f"_", "") for col in df.columns]

    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df.rename(columns={'Adj Close': 'Adj_Close'})

    base_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if 'Adj_Close' in df.columns:
        base_cols.append('Adj_Close')

    df = df[[col for col in base_cols if col in df.columns]]
    df['ticker'] = ticker

    # 🌍 Convert to USD if needed
    if '-' in ticker and not ticker.endswith('USD'):
        currency = ticker.split('-')[-1]
        fx_ticker = f"{currency}=X"
        print(f"🌐 Downloading FX rate: {fx_ticker} for conversion to USD...")
        fx = yf.download(fx_ticker, interval=interval, start=start)

        if fx.empty or 'Close' not in fx:
            print(f"⚠️ FX rate data unavailable for {fx_ticker}. Skipping conversion.")
        else:
            fx = fx[['Close']].reset_index()
            fx.columns = ['Date', 'FX_Close']
            fx['Date'] = pd.to_datetime(fx['Date']).dt.strftime('%Y-%m-%d %H:%M:%S')
            df = df.merge(fx, on='Date', how='left')

            for col in ['Open', 'High', 'Low', 'Close', 'Adj_Close']:
                if col in df.columns:
                    df[col] = df[col] / df['FX_Close']
            df.drop(columns=['FX_Close'], inplace=True)
            print(f"💱 Converted {ticker} OHLC data to USD using {fx_ticker}")

    filename = f"{ticker.replace('-', '_').replace('^', '')}_history_{execution_time}.parquet"
    print(f"✅ Starting upload file process: {filename} → {stage_area}")
    upload_file(df=df, stage_area=stage_area, private_key=private_key, filename=filename)
    print(f"✅ Upload complete for {ticker} → {stage_area}")




def technical_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """
    Receives a DataFrame called 'data' and returns it with technical indicators and corresponding categorical signals.

    Parameters:
        data (pd.DataFrame): Input DataFrame containing 'open', 'high', 'low', 'close', 'volume'

    Returns:
        pd.DataFrame: DataFrame with added technical indicators and signal columns
    """
    import ta
    import numpy as np

    if data.empty:
        print("⚠️ Received an empty DataFrame.")
        return pd.DataFrame()

    # Ensure required columns exist
    required_cols = ['high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Missing required column: '{col}'")

    # MACD
    macd_object = ta.trend.MACD(data['close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
    data['MACD'] = macd_object.macd()
    data['MACD_Signal'] = macd_object.macd_signal()
    data['MACD_Diff'] = macd_object.macd_diff()
    data['MACD_Signal_Label'] = np.where(data['MACD_Diff'] > 0, 'buy', np.where(data['MACD_Diff'] < 0, 'sell', 'neutral'))

    # MFI
    mfi_indicator = ta.volume.MFIIndicator(
        high=data['high'], low=data['low'], close=data['close'], volume=data['volume'], window=14, fillna=True
    )
    data['MFI'] = mfi_indicator.money_flow_index()
    data['MFI_Signal_Label'] = np.where(data['MFI'] > 80, 'sell', np.where(data['MFI'] < 20, 'buy', 'neutral'))

    # RSI
    data['RSI'] = ta.momentum.RSIIndicator(data['close'], window=14, fillna=True).rsi()
    data['RSI_Signal_Label'] = np.where(data['RSI'] > 70, 'sell', np.where(data['RSI'] < 30, 'buy', 'neutral'))

    # EMA
    data['EMA_Short'] = ta.trend.EMAIndicator(data['close'], window=12, fillna=True).ema_indicator()
    data['EMA_Long'] = ta.trend.EMAIndicator(data['close'], window=26, fillna=True).ema_indicator()
    data['EMA_Trend_Label'] = np.where(data['EMA_Short'] > data['EMA_Long'], 'buy', np.where(data['EMA_Short'] < data['EMA_Long'], 'sell', 'neutral'))

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(data['close'], window=20, window_dev=2, fillna=True)
    data['Bollinger_Upper'] = bb.bollinger_hband()
    data['Bollinger_Lower'] = bb.bollinger_lband()
    data['Bollinger_Middle'] = bb.bollinger_mavg()
    data['Bollinger_Signal_Label'] = np.where(data['close'] < data['Bollinger_Lower'], 'buy', np.where(data['close'] > data['Bollinger_Upper'], 'sell', 'neutral'))

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(
        high=data['high'], low=data['low'], close=data['close'], window=14, smooth_window=3, fillna=True
    )
    data['Stochastic'] = stoch.stoch()
    data['Stochastic_Signal'] = stoch.stoch_signal()
    data['Stochastic_Signal_Label'] = np.where(data['Stochastic'] > 80, 'sell', np.where(data['Stochastic'] < 20, 'buy', 'neutral'))

    # ATR
    #atr = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close'], window=14, fillna=True)
    #data['ATR'] = atr.average_true_range()

    # Parabolic SAR
    psar = ta.trend.PSARIndicator(data['high'], data['low'], data['close'], fillna=True)
    data['SAR'] = psar.psar()
    data['SAR_Signal_Label'] = np.where(data['close'] > data['SAR'], 'buy', np.where(data['close'] < data['SAR'], 'sell', 'neutral'))

    print(f"Technical Analysis function has been finished")

    return data


def generate_candlestick_patterns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds TA-Lib candlestick pattern columns and a 'pattern_label' column to the DataFrame.
    Optimized to avoid DataFrame fragmentation warnings.
    """
    if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        raise ValueError("Missing required OHLC columns.")

    import talib

    # Get all TA-Lib candlestick pattern function names
    candle_names = talib.get_function_groups()['Pattern Recognition']
    patterns = {name: getattr(talib, name) for name in candle_names}

    # Store all new columns here before merging
    pattern_cols = {}

    # Initialize pattern_label
    pattern_label = pd.Series(['none'] * len(data), index=data.index)

    for name, func in patterns.items():
        signal = func(data['Open'], data['High'], data['Low'], data['Close'])
        bull_col = name + '_bull'
        bear_col = name + '_bear'

        pattern_cols[bull_col] = (signal > 0).astype(int)
        pattern_cols[bear_col] = (signal < 0).astype(int)

        # Update pattern_label only for first match
        pattern_label = pattern_label.mask(
            (pattern_label == 'none') & (signal > 0), name + '_bull'
        )
        pattern_label = pattern_label.mask(
            (pattern_label == 'none') & (signal < 0), name + '_bear'
        )

    # Merge new columns into the main DataFrame
    data = pd.concat([data, pd.DataFrame(pattern_cols, index=data.index)], axis=1)
    data['pattern_label'] = pattern_label

    print(f"Generating Candlestick Patterns Function has been finished")

    return data

import requests
import pandas as pd
from datetime import datetime

# List of desired metrics
BLOCKCHAIN_METRICS = {
    "hash-rate": "hash_rate",
    "miners-revenue": "miner_revenue",
    "n-unique-addresses": "active_addresses",
    "bitcoin-days-destroyed": "days_destroyed",
    "difficulty": "difficulty"
}

BASE_URL = "https://api.blockchain.info/charts/"


def get_blockchain_metric(metric_name: str, timespan: str = "all") -> pd.DataFrame:
    """
    Downloads historical daily data from blockchain.com charts API.

    Args:
        metric_name (str): The API chart metric name (e.g., 'hash-rate')
        timespan (str): Time span of data to fetch (default: 'all')

    Returns:
        pd.DataFrame: DataFrame with timestamp and metric value
    """
    url = f"{BASE_URL}{metric_name}?timespan={timespan}&format=json"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()["values"]

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["x"], unit="s")
    df = df.set_index("timestamp").rename(columns={"y": BLOCKCHAIN_METRICS[metric_name]})
    return df.drop(columns=["x"])


def fetch_all_blockchain_metrics() -> pd.DataFrame:
    """
    Fetches all defined metrics and merges them into a single DataFrame.

    Returns:
        pd.DataFrame: Combined DataFrame with all selected metrics
    """
    dfs = []
    for api_name in BLOCKCHAIN_METRICS:
        print(f"Fetching {api_name}...")
        df = get_blockchain_metric(api_name)
        dfs.append(df)

    # Merge on timestamp index
    df_merged = pd.concat(dfs, axis=1).sort_index()
    return df_merged


# Example usage:
# df_blockchain = fetch_all_blockchain_metrics()
# df_blockchain.to_csv("blockchain_daily_metrics.csv")


def upsert_timeseries_to_snowflake(df, table_name, engine, schema='PUBLIC', database='CRYPTONETWORKANALYSIS'):
    """
    Appends new rows to a Snowflake table if they don't already exist (based on Date column).
    Creates the table if it does not exist.

    Parameters:
        df (pd.DataFrame): DataFrame with a 'Date' column.
        table_name (str): Target table name in Snowflake.
        engine: SQLAlchemy engine for Snowflake.
        schema (str): Schema name (default is 'PUBLIC').
        database (str): Database name (default is 'CRYPTODB').
    """

    with engine.connect() as conn:
        existing_dates = set()

        # Step 1: Check if table exists
        result = conn.execute(text(f"""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = '{schema}' AND table_name = '{table_name.upper()}'
        """)).fetchone()

        if result is not None:
            print(f"📌 Table {table_name.upper()} exists. Checking for existing data...")
            query = f'SELECT DISTINCT "Date" FROM {database}.{schema}."{table_name.upper()}"'
            date_rows = conn.execute(text(query)).fetchall()
            existing_dates = {row[0].date() for row in date_rows}
        else:
            print(f"📌 Table {table_name.upper()} does not exist. It will be created.")

    # Step 2: Filter only new rows
    df['Date_only'] = df['Date'].dt.date
    new_data = df[~df['Date_only'].isin(existing_dates)].drop(columns='Date_only')

    # Step 3: Insert new rows if needed
    if not new_data.empty:
        print(f"🆕 Appending {len(new_data)} new rows to {table_name.upper()}...")
        new_data.to_sql(
            name=table_name,
            con=engine,
            schema=schema,
            if_exists='append',
            index=False,
            dtype={
                "Date": DateTime(),
                "Open": Float(),
                "High": Float(),
                "Low": Float(),
                "Close": Float(),
                "Volume": Float()
            }
        )
    else:
        print(f"✅ No new data to append — Snowflake table {table_name.upper()} is up-to-date.")



import requests

def get_cmc_crypto_data(ticker: str, start: str, end: str, interval='daily', api_key=None):
    symbol, fiat = ticker.split('-')
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"

    params = {
        "symbol": symbol,
        "convert": fiat,
        "time_start": start,
        "time_end": end,
        "interval": interval
    }

    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": os.getenv("COINMARKETCAP")
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print(f"⚠️ CMC API Error: {response.status_code} – {response.text}")
        return pd.DataFrame()

    raw = response.json()
    quotes = raw.get('data', {}).get('quotes', [])
    if not quotes:
        print(f"⚠️ No data from CMC for {ticker}")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame([{
        'Date': q['timestamp'],
        'Open': q['quote'][fiat]['open'],
        'High': q['quote'][fiat]['high'],
        'Low': q['quote'][fiat]['low'],
        'Close': q['quote'][fiat]['close'],
        'Volume': q['quote'][fiat]['volume']
    } for q in quotes])
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['ticker'] = ticker
    return df


import requests
import pandas as pd

def get_coingecko_crypto_data(ticker: str, start: str, end: str, interval: str = 'daily') -> pd.DataFrame:
    symbol, fiat = ticker.split('-')
    coin_id_map = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'LTC': 'litecoin',
        # Add more mappings here
    }

    coin_id = coin_id_map.get(symbol.upper())
    if not coin_id:
        print(f"❌ Coin ID not mapped for symbol: {symbol}")
        return pd.DataFrame()

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    delta_days = (end_dt - start_dt).days

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': fiat.lower(),
        'days': delta_days,
        'interval': interval
    }

    print(f"📥 Fetching {symbol}-{fiat} from CoinGecko ({delta_days} days)...")
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print("⚠️ Error fetching CoinGecko data:", response.text)
        return pd.DataFrame()

    data = response.json()
    prices = data.get('prices', [])

    df = pd.DataFrame(prices, columns=['timestamp', 'Close'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)]

    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['ticker'] = f"{symbol.upper()}-{fiat.upper()}"

    return df[['Date', 'Close', 'ticker']]




def kmeans_cluster_from_price_data(df, value_col='close', min_k=2, max_k=10, plot=False):
    # 1. Pivot: rows = dates, columns = tickers, values = close prices
    price_df = df.pivot(index='date', columns='ticker', values=value_col)
    return_df = price_df.pct_change().dropna()

    # 2. Transpose and scale (tickers = rows)
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(return_df.T)

    # 3. Elbow method (SSE)
    sse = []
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(scaled_returns)
        sse.append(kmeans.inertia_)

    elbow_k = np.argmin(np.diff(sse)) + min_k

    # 4. Silhouette method
    sil_scores = []
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(scaled_returns)
        score = silhouette_score(scaled_returns, labels)
        sil_scores.append(score)

    sil_k = np.argmax(sil_scores) + min_k

    # 5. Average best cluster count
    best_k = round((elbow_k + sil_k) / 2)

    # 6. Final model
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
    labels = final_kmeans.fit_predict(scaled_returns)

    # 7. Output
    result = pd.DataFrame({
        'ticker': return_df.columns,
        'cluster': labels
    })

    # Optional: show plots
    if plot:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(min_k, max_k + 1), sse, marker='o')
        plt.title('Elbow Method (SSE)')
        plt.xlabel('k')
        plt.ylabel('SSE')

        plt.subplot(1, 2, 2)
        plt.plot(range(min_k, max_k + 1), sil_scores, marker='o')
        plt.title('Silhouette Scores')
        plt.xlabel('k')
        plt.ylabel('Silhouette Score')

        plt.tight_layout()
        plt.show()

    print(f"📌 Final number of clusters used: {best_k}")
    return result

def cluster_from_correlation(df, value_col='close', method='complete', k=4, plot=False):

    df = df.tail(df.shape[0] - 1)
    price_df = df.pivot(index='date', columns='ticker', values=value_col)
    #price_df = price_df.replace(0, np.nan).dropna(axis=1, thresh=int(0.85 * len(price_df)))
    price_df = price_df.ffill().bfill().fillna(price_df.mean())

    returns = np.log(price_df / price_df.shift(1)).fillna(0)
    
    # Step 2: Correlation + distance matrix
    corr = returns.corr()
    dist = np.sqrt(2 * (1 - corr))

    

    # Step 3: Agglomerative clustering
    model = AgglomerativeClustering(
        metric='precomputed',
        linkage=method,  # use 'average' or 'complete'
        n_clusters=k
    )
    
    labels = model.fit_predict(dist)

    # Step 4: Build result
    cluster_df = pd.DataFrame({
        'ticker': dist.columns,
        'cluster': labels
    })

    # Optional: plot heatmap + dendrogram
    if plot:
        sns.clustermap(corr, cmap='coolwarm', figsize=(12, 10), metric='euclidean',
                       method=method, row_cluster=True, col_cluster=True)
        plt.title('Hierarchical Clustering Heatmap')
        plt.show()

    return cluster_df


def louvain_from_returns(df, value_col='close', min_corr=0.5, plot=True, show_mixed_only=False):
    import numpy as np
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
    import community  # python-louvain
    from matplotlib.cm import get_cmap

    # Step 1: Pivot + fill gaps due to mixed trading calendars
    price_df = df.pivot(index='date', columns='ticker', values=value_col)
    price_df = price_df.ffill().bfill().fillna(price_df.mean())

    # Optional: print tickers being used
    print("✅ Tickers included after fill:", price_df.columns.tolist())

    # Step 2: Compute log returns
    returns = np.log(price_df / price_df.shift(1)).dropna()

    # Step 3: Correlation matrix
    corr = returns.corr()

    # Step 4: Build correlation graph
    G = nx.Graph()
    for i in corr.columns:
        for j in corr.columns:
            if i != j and corr.loc[i, j] > min_corr:
                G.add_edge(i, j, weight=corr.loc[i, j], distance=1 - corr.loc[i, j])

    # Step 5: Louvain clustering
    partition = community.best_partition(G, weight='weight')
    nx.set_node_attributes(G, partition, 'cluster')
    cluster_df = pd.DataFrame.from_dict(partition, orient='index', columns=['cluster']).reset_index().rename(columns={'index': 'ticker'})

    # Print clusters
    print("📊 Louvain Clusters:")
    for cid in set(partition.values()):
        members = [ticker for ticker, c in partition.items() if c == cid]
        has_crypto = any('-USD' in m or m.startswith('BTC') for m in members)
        has_stock = any('-USD' not in m and not m.startswith('BTC') for m in members)
        if (not show_mixed_only) or (has_crypto and has_stock):
            print(f"\n🧠 Cluster {cid} ({'MIXED' if has_crypto and has_stock else 'CRYPTO' if has_crypto else 'STOCK'}):")
            print(members)

    # Filter returns to clustered tickers
    clustered_returns = returns.copy()
    clustered_returns.columns.name = None
    clustered_returns = clustered_returns.loc[:, clustered_returns.columns.intersection(cluster_df['ticker'])]

    # Optional: Plot network
    if plot:
        pos = nx.spring_layout(G, seed=42, k=0.3)
        node_colors = ['blue' if '-USD' in n or n.startswith('BTC') else 'orange' for n in G.nodes()]
        cmap = get_cmap('tab10')

        plt.figure(figsize=(14, 10))
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors)
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        nx.draw_networkx_labels(G, pos, font_size=9)
        plt.title(f"Louvain Correlation Graph (min_corr={min_corr})")
        plt.axis('off')
        plt.show()

    # Print unconnected nodes
    unconnected = [n for n in corr.columns if n not in G.nodes]
    print(f"🪫 Unconnected tickers ({len(unconnected)}):", unconnected)

    return cluster_df, G, clustered_returns





def plot_ticker_neighborhood(G, central_node='BTC-USD', hops=2):
    # Step 1: Get all nodes within `hops` from BTC-USD
    btc_subgraph_nodes = nx.single_source_shortest_path_length(G, central_node, cutoff=hops).keys()
    
    # Step 2: Extract subgraph
    subG = G.subgraph(btc_subgraph_nodes)

    # Optional: re-run Louvain just on the subgraph
    partition = community.best_partition(subG, weight='weight')
    nx.set_node_attributes(subG, partition, 'cluster')

   
    # Step 3: Visualize
    pos = nx.spring_layout(subG, seed=42, k=0.3)
    cmap = plt.get_cmap('tab10')

    
    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(subG, pos, node_color=[partition[n] for n in subG.nodes()],
                           node_size=600, cmap=cmap)
    nx.draw_networkx_edges(subG, pos, alpha=0.4)
    nx.draw_networkx_labels(subG, pos, font_size=9)
    
    plt.title(f'Louvain Subgraph: Nodes within {hops} hops from {central_node}')
    plt.axis('off')
    plt.show()

