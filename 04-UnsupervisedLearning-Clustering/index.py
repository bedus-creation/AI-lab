from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


X, y_true = make_blobs(n_samples=500, centers=4,
                       cluster_std=0.40, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.show()
