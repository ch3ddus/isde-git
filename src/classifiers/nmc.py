import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class NMC(object):
    """
    Class implementing the Nearest Mean Centroid (NMC) classifier.

    This classifier estimates one centroid per class from the training data,
    and predicts the label of a never-before-seen (test) point based on its
    closest centroid.

    Attributes
    -----------------
    - centroids: read-only attribute containing the centroid values estimated
        after training

    Methods
    -----------------
    - fit(x,y) estimates centroids from the training data
    - predict(x) predicts the class labels on testing points

    """

    def __init__(self):
        self._centroids = None
        self._class_labels = None  # class labels may not be contiguous indices

    @property
    def centroids(self):
        return self._centroids

    @property
    def class_labels(self):
        return self._class_labels

    def fit(self, xtr, ytr):
        """
        Calcola il centroide medio per ciascuna classe e popola self._centroids.
        """
        self._classes = np.unique(y)
        num_classes = len(self._classes)
        num_features = x.shape[1]
        
        self._centroids = np.zeros((num_classes, num_features))
        for i, c in enumerate(self._classes):
            self._centroids[i] = np.mean(x[y == c], axis=0)

    def predict(self, xts):
        """
        Predice la classe per ciascun campione in ingresso.
        """
        distances = np.linalg.norm(x[:, np.newaxis, :] - self._centroids[np.newaxis, :, :], axis=2)
        nearest_indices = np.argmin(distances, axis=1)
        return self._classes[nearest_indices]
