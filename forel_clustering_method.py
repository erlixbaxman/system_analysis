import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
data = iris.data


def forel_clustering(data, radius):
    # Initialize list of clusters
    clusters = []

    # Iterate over all data points
    for point in data:
        # Check if point belongs to any existing cluster
        assigned = False
        for cluster in clusters:
            if np.linalg.norm(point - cluster[0]) <= radius:
                cluster.append(point)
                assigned = True
                break

        # If point does not belong to any existing cluster, create a new cluster
        if not assigned:
            clusters.append([point])

    return clusters

# Perform Forel clustering on the Iris dataset
clusters = forel_clustering(data, 2)

# Plot the clusters
plt.figure(figsize=(8, 6))
for i, cluster in enumerate(clusters):
    cluster = np.array(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i+1}')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.legend()
plt.show()
