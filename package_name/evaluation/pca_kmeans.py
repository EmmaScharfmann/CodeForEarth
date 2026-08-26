from dataclasses import dataclass

import numpy as np
import joblib
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAKmeansModels:
    scaler: StandardScaler
    pca: PCA
    kmeans: KMeans


@dataclass
class PCAKmeansInfo:
    models: PCAKmeansModels
    explained_variance_ratio: np.ndarray


class PCAKmeansTrainer:
    """
    Trainer for the baseline model PCA + Kmeans.

    Attributes
    ----------
    n_clusters : int
        Number of clusters for K-means.
    n_components : int | None. Default None.
        Optional: Number of principal components. If None, uses min(n_samples, n_features)
    random_state : int
        Random seed for reproducibility. Default=42
    model_dir : str
        Directory path where models will be saved. Default: "./results/"

    Methods
    -------
    cluster():
        Perform PCA followed by K-means clustering with probability outputs.
    """

    def __init__(
        self,
        n_clusters: int,
        n_components: int | None = None,
        random_state: int = 42,
        model_dir: str = "./results/",
    ) -> None:
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.random_state = random_state
        self.model_dir = model_dir

    def cluster(self, x: np.ndarray) -> tuple[np.ndarray, PCAKmeansInfo]:
        """
        Perform PCA followed by K-means clustering with probability outputs.

        :param x:               ndarray of shape (n_samples, n_features). Input data matrix
        :return:                cluster_probs: Array of shape (n_clusters, n_samples), which contains
                                the probability of each sample of belonging to each cluster.
                                The probabilities for each sample sum to 1.0
                                model_info: dictionary with additional information on the pca kmeans model.
        """
        scaler, x_scaled = _fit_scaler(x=x)
        pca, x_pca = self._perform_pca(x=x_scaled)
        kmeans, cluster_probs = self._perform_kmeans_clustering(x=x_pca)

        models = PCAKmeansModels(
            pca=pca,
            kmeans=kmeans,
            scaler=scaler,
        )
        model_info = PCAKmeansInfo(
            models=models,
            explained_variance_ratio=pca.explained_variance_ratio_,
        )

        return cluster_probs, model_info

    def _perform_pca(self, x: np.ndarray) -> tuple[PCA, np.ndarray]:
        """Perform PCA with self.n_components on the given input."""
        n_components = (
            min(x.shape[0], x.shape[1])
            if self.n_components is None
            else self.n_components
        )
        pca = PCA(n_components=n_components, random_state=self.random_state)
        x_pca = pca.fit_transform(X=x)
        return pca, x_pca

    def _perform_kmeans_clustering(self, x: np.ndarray) -> tuple[KMeans, np.ndarray]:
        """Perform K-means clustering on the given input with self.n_clusters."""
        kmeans = KMeans(
            n_init=20,
            max_iter=300,
            n_clusters=self.n_clusters,
            random_state=self.random_state,
        )
        kmeans.fit(X=x)
        cluster_probs = _get_probabilities_from_kmeans(kmeans=kmeans, x=x)
        return kmeans, cluster_probs

    def save_models(self, info: PCAKmeansInfo) -> str:
        """
        Save trained PCA, K-means, and Scaler models to disk.

        :param info:            PCAKmeansInfo object containing 'pca_model', 'kmeans_model', and 'scaler_model'
        :return:                Directory of the saved models (Scaler, PCA, & KMeans)

        """
        model_dir = self.model_dir
        os.makedirs(model_dir, exist_ok=True)

        scaler_path = os.path.join(model_dir, "scaler.pkl")
        pca_path = os.path.join(model_dir, "pca.pkl")
        kmeans_path = os.path.join(model_dir, "kmeans.pkl")

        joblib.dump(info.models.scaler, scaler_path)
        joblib.dump(info.models.pca, pca_path)
        joblib.dump(info.models.kmeans, kmeans_path)

        return model_dir


class PCAKmeansPredictor:
    """
    Predictor for the baseline model PCA + Kmeans.

    Attributes
    ----------
    model_dir : str
        Directory path where models will be saved. Default: "./results/"
    model : PCAKmeansModels
        The models (scaler, PCA, Kmeans) loaded from the directory.

    Methods
    -------
    load_models():
        Load trained PCA, K-means, and Scaler models from self.model_dir.
    cluster():
        Predict cluster assignments and probabilities using the self.model.
    """

    def __init__(
        self,
        model_dir: str = "./results/",
    ) -> None:
        self.model_dir = model_dir
        self.model = self.load_models()

    def load_models(self) -> PCAKmeansModels:
        """
        Load trained PCA, K-means, and Scaler models.

        :return:                Object with loaded models
        """
        model_dir = self.model_dir

        scaler_path = os.path.join(model_dir, "scaler.pkl")
        pca_path = os.path.join(model_dir, "pca.pkl")
        kmeans_path = os.path.join(model_dir, "kmeans.pkl")

        scaler = joblib.load(scaler_path)
        pca = joblib.load(pca_path)
        kmeans = joblib.load(kmeans_path)

        return PCAKmeansModels(scaler=scaler, pca=pca, kmeans=kmeans)

    def clusters(self, x: np.ndarray):
        """
        Predict cluster assignments and probabilities for new data using trained models.
        New data must have the same number of features as training data.

        :param x:               ndarray of shape (n_samples, n_features) or (n_samples, n_members, n_features). New input data matrix
        :return:                cluster_probs: Array of shape (n_clusters, n_samples), which contains
                                the probability of each sample of belonging to each cluster.
        """
        models = self.model
        scaler = models.scaler
        
        if len(x.shape) == 2:
            x_scaled = scaler.transform(X=x)
            x_pca = models.pca.transform(X=x_scaled)
            cluster_probs = _get_probabilities_from_kmeans(x=x_pca, kmeans=models.kmeans)
        
        else:
            cluster_probs_mem = np.zeros((models.kmeans.n_clusters, x.shape[0], x.shape[1]))
            for member in range(x.shape[1]):
                x_scaled = scaler.transform(X=x[:, member, :])
                x_pca = models.pca.transform(X=x_scaled)
                cluster_probs_mem[:, :,member] = _get_probabilities_from_kmeans(x=x_pca, kmeans=models.kmeans)
            cluster_probs = np.mean(cluster_probs_mem, axis=-1)

        return cluster_probs


def _fit_scaler(x: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
    """Fit StandardScaler on input data."""
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X=x)
    return scaler, x_scaled


def _get_probabilities_from_kmeans(kmeans: KMeans, x: np.ndarray) -> np.ndarray:
    """Calculate the probability of being in a cluster from the cluster's distance.
    Instead of taking the closest cluster, we transform distances into probabilities and
    returns an array of probabilities for each cluster."""
    distances = kmeans.transform(X=x)
    similarities = np.exp(-distances)
    probabilities = similarities / similarities.sum(axis=1, keepdims=True)
    return probabilities.T
