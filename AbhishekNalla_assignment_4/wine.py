from sklearn.datasets import load_wine
from sklearn import cluster
import numpy as np

wine = load_wine()
test_idx = [10,100,150]

train_target = np.delete(wine.target, test_idx)
train_data = np.delete(wine.data, test_idx, axis=0)

test_target = wine.target[test_idx]
test_data = wine.data[test_idx]

k_means = cluster.KMeans(n_clusters = 2)
k_means.fit(train_data)
print("1. Predicted:",k_means.labels_[::10])
print("1. Expected:",train_target[::10])
print("\n")

k_means = cluster.KMeans(n_clusters = 3)
k_means.fit(train_data)
print("2. Predicted:",k_means.labels_[::10])
print("2. Expected:",train_target[::10])
print("\n")

k_means = cluster.KMeans(n_clusters = 4)
k_means.fit(train_data)
print("3. Predicted:",k_means.labels_[::10])
print("3. Expected:",train_target[::10])
