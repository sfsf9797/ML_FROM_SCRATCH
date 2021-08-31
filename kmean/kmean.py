import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import euclidean


class MyKMeans():

  def __init__(self, n_clusters, max_iters=350, random_state=0):
    np.random.seed(random_state)
    self.n_clusters = n_clusters
    self.max_iters = max_iters
    self.clusters = [[] for _ in range(self.n_clusters)]
    self.labels = None
    self.cluster_centers_ = []

  def fit(self, X):

    if not isinstance(X, np.ndarray):
      X = np.array(X)
    
    self.n_samples, self.n_features = X.shape

    if self.n_samples == 0:
      raise ValueError('The array is empty')
    
    self.X = X 
    self._fit(self.X)

    self.labels = np.zeros(self.n_samples)

    for cluster , indices in enumerate(self.clusters):
      for idx in indices:
        self.labels[idx] = cluster


    return self

  def predict(self, X):
    if not isinstance(X, np.ndarray):
      X = np.array(X)
    
    assert X.shape[1] == self.n_features ,'The number of features is not equal'

    pred = []

    for x in X:
      closest_idx = None
      shortest_dist = None

      for i, center in enumerate(self.cluster_centers_):
        dist = euclidean(x,center)
        if closest_idx is None or dist < shortest_dist:
          closest_idx = i
          shortest_dist = dist 
      pred.append(closest_idx)
    
    return pred

  def _fit(self, x):
    self.init_cluster_centers()
    cluster_centers = self.cluster_centers_ 

    for i in range(self.max_iters):
      self.assign_cluster_centers(cluster_centers)
      old_cluster_centers = cluster_centers
      cluster_centers = [self.get_cluster_center(cluster) for cluster in self.clusters]

      if self.is_converged(old_cluster_centers, cluster_centers):
        break
    self.cluster_centers_ = cluster_centers

  def init_cluster_centers(self):
    #random select k samples from training set
    self.cluster_centers_ = [self.X[idx] for idx in np.random.choice(range(self.n_samples), self.n_clusters)]

  def assign_cluster_centers(self, centers):
    #assign each data sample to a new center
    for sample in range(self.n_samples):
      for idx, center in enumerate(self.clusters):
        if sample in center:
          self.clusters[idx].remove(sample)
          break 
      
      closest = self._dist(sample, centers)
      self.clusters[closest].append(sample)

  def _dist(self, point, centers):
    #compute the eucledian distance between each data and the centroids / cluster center
    closest_idx = None
    shortest_dist = None

    for i, center in enumerate(centers):
      dist = euclidean(self.X[point],center)
      if closest_idx is None or dist < shortest_dist:
        closest_idx = i
        shortest_dist = dist 

    return closest_idx 

  def is_converged(self, old_cluster_centers, cluster_centers):
    # is converged when old centriods equal to centriods
    dist = 0
    for k in range(self.n_clusters):
      dist += euclidean(old_cluster_centers[k],cluster_centers[k])
    return dist == 0

    
  def get_cluster_center(self, center):
    cluster_center = []
    for i in range(self.n_features):
      cluster_center.append(np.mean(np.take(self.X[:,i], center)))

    return cluster_center

  def visualize(self):
      palette = sns.color_palette("husl", self.n_clusters + 1)
      data = self.X

      fig, ax = plt.subplots()

      for i, idx in enumerate(self.clusters):
          sample = np.array(data[idx]).T
          ax.scatter(*sample, c=[palette[i], ])

      for center in self.cluster_centers_:
          ax.scatter(*center, marker="x", linewidths=15)
    
      plt.show()
 
