import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("customer_data.csv")
features = ["Age","AnnualIncome","PurchaseFrequency","AvgOrderValue",
            "WebVisitsPerMonth","DiscountUsagePct","RecencyDays","SpendingScore"]

for col in features:
    df[col] = df[col].fillna(df[col].median())

X = StandardScaler().fit_transform(df[features])

scores = {}
models = {}
for k in range(2, 9):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    scores[k] = silhouette_score(X, labels)
    models[k] = model

best_k = max(scores, key=scores.get)
model = models[best_k]
df["Cluster"] = model.labels_

print("Best K:", best_k)
print("Best Silhouette Score:", round(scores[best_k], 3))
print("\nCluster sizes:")
print(df["Cluster"].value_counts().sort_index())

df.to_csv("customer_segments.csv", index=False)

plt.figure(figsize=(8,5))
plt.plot(list(scores.keys()), list(scores.values()), marker="o")
plt.title("Silhouette Score by Number of Clusters")
plt.xlabel("K"); plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig("outputs/silhouette_scores.png", dpi=160)
plt.show()
