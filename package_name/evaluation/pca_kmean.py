from dataclasses import dataclass

import numpy as np
import joblib
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAKmeansInfo:
    pca_model: PCA
    kmeans_model: KMeans
    scaler_model: StandardScaler
    explained_variance_ratio: np.ndarray


@dataclass
class PCAKmeansModelsPath:
    scaler_path: str
    pca_path: str
    kmeans_path: str


@dataclass
class PCAKmeansModels:
    scaler: StandardScaler
    pca: PCA
    kmeans: KMeans


def pca_kmeans_clustering(
    x: np.ndarray, n_clusters: int, n_components: int | None = None, random_state=42
) -> tuple[list[np.ndarray], PCAKmeansInfo]:
    """
    Perform PCA followed by K-means clustering with probability outputs.

    :param x:               ndarray of shape (n_samples, n_features). Input data matrix
    :param n_clusters:      Number of clusters for K-means
    :param n_components:    Optional: Number of principal components. If None, uses min(n_samples, n_features)
    :param random_state:    Random seed for reproducibility. Default=42
    :return:                cluster_probs: List of length n_clusters, where each element is an array of shape (n_samples,)
                            containing the probability of each sample belonging to that cluster.
                            The probabilities for each sample sum to 1.0
                            model_info: dictionary with additional information on the pca kmeans model.
    """
    scaler, x_scaled = _fit_scaler(x=x)
    pca, x_pca = _perform_pca(
        x=x_scaled, n_components=n_components, random_state=random_state
    )
    kmeans, cluster_probs = _perform_k_mean_clustering(
        x=x_pca, n_clusters=n_clusters, random_state=random_state
    )

    model_info = PCAKmeansInfo(
        pca_model=pca,
        kmeans_model=kmeans,
        scaler_model=scaler,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )

    return cluster_probs, model_info


def _fit_scaler(x: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
    """
    Fit StandardScaler on input data.

    :param x:               Input data matrix of shape (n_samples, n_features).
    :return:                Tuple with scaler: Fitted StandardScaler object and
                            x_scaled: Scaled data of shape (n_samples, n_features)
    """
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X=x)
    return scaler, x_scaled


def _perform_pca(
    x: np.ndarray, n_components: int | None = None, random_state=42
) -> tuple[PCA, np.ndarray]:
    """
    Perform PCA with n_components on the given input.

    :param x:               ndarray of shape (n_samples, n_features). Scaled input data matrix
    :param n_components:    Optional: Number of principal components. If None, uses min(n_samples, n_features)
    :param random_state:    Random seed for reproducibility. Default=42
    :return:                Tuple with X_pca: PCA-transformed data of shape (n_samples, n_components) and
                            pca: Fitted PCA object
    """
    if n_components is None:
        n_components = min(x.shape[0], x.shape[1])

    pca = PCA(n_components=n_components, random_state=random_state)
    x_pca = pca.fit_transform(X=x)
    return pca, x_pca


def _perform_k_mean_clustering(
    x: np.ndarray, n_clusters: int, random_state: int = 42
) -> tuple[KMeans, list[np.ndarray]]:
    """
    Perform K-means clustering on the given input with n_clusters.

    :param x:               Input data matrix PCA-transformed of shape (n_samples, n_components).
    :param n_clusters:      Number of clusters for K-means
    :param random_state:    Random seed for reproducibility. Default=42
    :return:                Tuple of cluster_probs: List of length n_clusters, where each element is an array of shape (n_samples,)
                            containing the probability of each sample belonging to that cluster and
                            kmeans: Fitted KMeans object
    """
    kmeans = KMeans(
        n_clusters=n_clusters, random_state=random_state, n_init=20, max_iter=300
    )
    distances = kmeans.fit_transform(X=x)
    cluster_probs = _get_probabilities_from_distances(
        distances=distances, n_clusters=n_clusters
    )
    return kmeans, cluster_probs


def _get_probabilities_from_distances(
    distances: np.ndarray, n_clusters: int
) -> list[np.ndarray]:
    """Calculate the probability of being in a cluster from the cluster's distance.
    Instead of taking the closest cluster, we transform distances into probabilities and
    returns the list of probabilities for each cluster."""
    similarities = np.exp(-distances)
    probabilities = similarities / similarities.sum(axis=1, keepdims=True)
    cluster_probs = [probabilities[:, i] for i in range(n_clusters)]
    return cluster_probs


def save_models(
    info: PCAKmeansInfo, model_dir: str = "./models/"
) -> PCAKmeansModelsPath:
    """
    Save trained PCA, K-means, and Scaler models to disk.

    :param info:            PCAKmeansInfo object containing 'pca_model', 'kmeans_model', and 'scaler_model'
    :param model_dir:       Directory path where models will be saved. Default: "./models/"
    :return:                Object with paths to saved models (Scaler, PCA, & KMeans)

    """
    os.makedirs(model_dir, exist_ok=True)

    scaler_path = os.path.join(model_dir, "scaler.pkl")
    pca_path = os.path.join(model_dir, "pca.pkl")
    kmeans_path = os.path.join(model_dir, "kmeans.pkl")

    joblib.dump(info.scaler_model, scaler_path)
    joblib.dump(info.pca_model, pca_path)
    joblib.dump(info.kmeans_model, kmeans_path)

    return PCAKmeansModelsPath(
        scaler_path=scaler_path, pca_path=pca_path, kmeans_path=kmeans_path
    )


def load_models(model_dir: str = "./models/") -> PCAKmeansModels:
    """
    Load trained PCA, K-means, and Scaler models from disk.

    :param model_dir:       Directory path where models are saved. Default: "./models/"
    :return:                Object with loaded models
    """
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    pca_path = os.path.join(model_dir, "pca.pkl")
    kmeans_path = os.path.join(model_dir, "kmeans.pkl")

    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    kmeans = joblib.load(kmeans_path)

    return PCAKmeansModels(scaler=scaler, pca=pca, kmeans=kmeans)


def predict_clusters(x: np.ndarray, models: PCAKmeansModels):
    """
    Predict cluster assignments and probabilities for new data using trained models.
    New data must have the same number of features as training data.

    :param x:               ndarray of shape (n_samples, n_features). New input data matrix
    :param models:          PCAKmeansModels object with 'scaler_model', 'pca_model', 'kmeans_model'
    :return:                cluster_probs: List of length n_clusters, where each element is an array of shape (n_samples,)
                            containing the probability of each sample belonging to that cluster.
    """
    scaler = models.scaler
    x_scaled = scaler.transform(X=x)

    pca = models.pca
    x_pca = pca.transform(X=x_scaled)

    cluster_probs = _predict_with_kmeans(x=x_pca, kmeans=models.kmeans)

    return cluster_probs


def _predict_with_kmeans(x: np.ndarray, kmeans: KMeans):
    """
    Predict cluster assignments and probabilities using a fitted K-means model.

    :param x:               ndarray of shape (n_samples, n_components). Input data matrix (typically PCA-transformed)
    :param kmeans:          Fitted KMeans object
    :return:                cluster_probs: List of length n_clusters, where each element is an array of shape (n_samples,)
                            containing the probability of each sample belonging to that cluster.
    """
    distances = kmeans.transform(x)
    cluster_probs = _get_probabilities_from_distances(
        distances=distances, n_clusters=kmeans.n_clusters
    )
    return cluster_probs
