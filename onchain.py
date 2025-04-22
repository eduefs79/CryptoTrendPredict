import requests
import pandas as pd
from datetime import datetime

    # List of desired metrics
BLOCKCHAIN_METRICS = {
    "hash-rate": "hash_rate",
    "miners-revenue": "miner_revenue",
    "n-unique-addresses": "active_addresses",
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





