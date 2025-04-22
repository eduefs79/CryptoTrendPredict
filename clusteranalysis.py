import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def perform_kmeans_with_elbow_silhouette(df: pd.DataFrame, plot: bool = True) -> pd.DataFrame:
    """
    Perform K-Means clustering using the average of Elbow and Silhouette methods to determine the best number of clusters.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        plot (bool): Whether to display the Elbow and Silhouette plots (default: True)

    Returns:
        pd.DataFrame: Original DataFrame with an added 'Cluster' column
    """
    # --- Step 1: Select numeric columns only, excluding datetime ---
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    df_numeric = df_numeric.dropna()


    # --- Step 3: Elbow Method ---
    sse = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_numeric)
        sse.append(kmeans.inertia_)

    # Elbow detection
    deltas = np.diff(sse, 2)
    elbow_k = k_range[np.argmin(deltas) + 1]

    # --- Step 4: Silhouette Scores ---
    silhouette_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(df_numeric)
        score = silhouette_score(scaled_data, labels)
        silhouette_scores.append(score)

    silhouette_k = k_range[np.argmax(silhouette_scores)]
    avg_k = round((elbow_k + silhouette_k) / 2)

    print(f"Elbow suggests: {elbow_k}, Silhouette suggests: {silhouette_k}, Using average: {avg_k}")

    # --- Step 5: Plotting ---
    if plot:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(k_range, sse, marker='o')
        plt.axvline(elbow_k, color='r', linestyle='--', label=f'Elbow at k={elbow_k}')
        plt.title("Elbow Method")
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("SSE")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(k_range, silhouette_scores, marker='o', color='green')
        plt.axvline(silhouette_k, color='r', linestyle='--', label=f'Best silhouette at k={silhouette_k}')
        plt.title("Silhouette Score")
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Score")
        plt.legend()

        plt.tight_layout()
        plt.show()

    # --- Step 6: Final Clustering ---
    final_kmeans = KMeans(n_clusters=avg_k, random_state=42)
    final_labels = final_kmeans.fit_predict(df_numeric)

    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = -1
    df_with_clusters.loc[df_numeric.index, 'Cluster'] = final_labels

    return df_with_clusters
