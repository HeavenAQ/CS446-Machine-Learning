import numpy as np


class KMeans:
    def __init__(self, n_clusters: int, max_iter: int):
        """Init the center of clusters.

        Parameters
        ----------
        n_clusters : number of clusters
        max_iter : number of interations


        Attribute
        -------
        cluster_centers_ : cluster centers
        max_iter: number of interations
        n_clusters: number of clusters
        """
        self.cluster_centers_ = []
        self.labels_ = []
        self.max_iter = max_iter
        self.n_clusters = n_clusters

    def _init_centers(self, X: np.ndarray) -> np.ndarray:
        """Init the center of clusters.

        Parameters
        ----------
        X : np.ndarray
            The training input samples.


        Returns
        -------
        Centers : np.ndarray [n_cluster,3]
            The cluster centers
        """
        idxs = np.random.choice(X.shape[0], self.n_clusters, replace=True)
        self.cluster_centers_ = X[idxs]
        return self.cluster_centers_

    def fit(self, X: np.ndarray):
        """Fit with Kmeans algorithm

        Parameters
        ----------
        X : np.ndarray
            The training input samples.


        Returns
        -------
        Centers : np.ndarray [n_cluster,3]
            The cluster centers
        """
        if len(self.cluster_centers_) == 0:
            self.cluster_centers_ = self._init_centers(X)

        N = X.shape[0]
        for _ in range(self.max_iter):
            # Assignment Step
            self.labels_ = np.empty(N, dtype=np.int64)
            for i in range(N):
                self.labels_[i] = np.argmin(
                    np.sum(
                        (X[i] - self.cluster_centers_) ** 2,
                        axis=1,
                    )
                )

            # Update Step
            new_centers = np.empty_like(self.cluster_centers_)
            for class_val in np.unique(self.labels_):
                new_centers[class_val] = X[self.labels_ == class_val].mean()

            if np.allclose(new_centers, self.cluster_centers_):
                self.cluster_centers_ = new_centers
                break

        return self.cluster_centers_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with Kmeans algorithm

        Parameters
        ----------
        X : np.ndarray
            The training input samples.

        Returns
        -------
        Labels : np.ndarray
                 Predicted labels
        """
        self.labels_ = np.asarray(
            [
                np.argmin(
                    np.sum(
                        (X[i] - self.cluster_centers_) ** 2,
                        axis=1,
                    )
                )
                for i in range(X.shape[0])
            ]
        )
        return self.labels_
