import pandas as pd
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("games.csv")

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print("Rows:", len(df))
df.head()

def convert_k(x):
    if isinstance(x, str) and x.endswith("K"):
        return float(x.replace("K", "")) * 1000
    try:
        return float(x)
    except:
        return np.nan

num_cols = ["Plays", "Playing", "Backlogs", "Wishlist", "Rating"]
for col in num_cols:
    df[col] = df[col].apply(convert_k)

# Remove rows missing too many numeric values
df = df.dropna(subset=num_cols)

df[num_cols].head()

# Convert genre-list strings into real Python lists
df["Genres"] = df["Genres"].apply(lambda g: " ".join(literal_eval(g)))

vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(df["Genres"])

genre_matrix.shape


scaler = StandardScaler()
num_scaled = scaler.fit_transform(df[num_cols])

from scipy.sparse import hstack
X = hstack([num_scaled, genre_matrix])

sil_scores = {}

for k in range(3, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels, metric="cosine")
    sil_scores[k] = score
    print(f"k={k}, silhouette={score:.4f}")

best_k = max(sil_scores, key=sil_scores.get)
best_k

#bar graph for silhouette scores
plt.figure(figsize=(10, 6))
bars = plt.bar(sil_scores.keys(), sil_scores.values(), color='skyblue', edgecolor='navy', alpha=0.7)

#best k value
best_score = sil_scores[best_k]
bars[list(sil_scores.keys()).index(best_k)].set_color('lightcoral')
for k, score in sil_scores.items():
    plt.text(k, score + 0.005, f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

plt.xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
plt.ylabel('Silhouette Score', fontsize=12, fontweight='bold')
plt.title('Silhouette Scores for Different k Values\n(Optimal k highlighted in red)', fontsize=14, fontweight='bold')
plt.xticks(list(sil_scores.keys()))
plt.grid(axis='y', alpha=0.3)

plt.axhline(y=best_score, color='red', linestyle='--', alpha=0.5, label=f'Best Score: {best_score:.4f}')
plt.legend()

plt.tight_layout()
plt.show()

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
df["cluster"] = kmeans.fit_predict(X)

df[["Title", "Genres", "Rating", "Plays", "cluster"]].head(20)

for c in sorted(df.cluster.unique()):
    print(f"\n=== Cluster {c} ===")
    print(df[df.cluster == c].head(5)[["Title", "Genres", "Rating", "Plays"]])

df.to_csv("games_clustered.csv", index=False)
print("Saved to /mnt/data/games_clustered.csv")
print()
# Example analysis you could provide:
cluster_summary = df.groupby('cluster').agg({
    'Rating': 'mean',
    'Plays': 'mean', 
    'Wishlist': 'mean',
    'Backlogs': 'mean'
}).round(2)

print("Cluster Performance Profiles:")
print(cluster_summary)