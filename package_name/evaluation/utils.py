import numpy as np
from package_name.inference.predictor import VAEPredictor
from package_name.evaluation.pca_kmeans import PCAKmeansPredictor
from package_name.data_processing.data_processor import flatten_input
from package_name.evaluation.metrics import (
    compute_conditional_probabilities,
    compute_BSS_clusters_target,
    compute_inputs_to_calc_quantile_exceedance,
)


def compute_BSS_clusters_target_from_pca(
    pca_kmeans_predictor: PCAKmeansPredictor,
    input_test: np.ndarray,
    targets_categorical_test: np.ndarray,
    input_climatology: np.ndarray = None,
    targets_categorical_climatology: np.ndarray = None,
) -> float:
    """
    Calculate the Brier skill score of test data for the classification of cluster assignments predicted by a PCA + Kmeans model. The conditional probabilities of the targets given the clusters and the baseline forecast (climatology) are computed from the training data, if provided. If no training data is provided this will be computed from the test data.

    :param pca_kmeans_predictor: The PCA + Kmeans model used to predict cluster assignments
    :param input_test: Test data for the input variable, with shape (times, latitudes, longitudes)
        or, for an ensemble, (times, members, latitudes, longitudes).
    :param targets_categorical_test: Test data for the target categories, with shape (times, ..., classes).
    :param input_climatology: Climatology data for the input variable, with shape (times, latitudes, longitudes)
        or, for an ensemble, (times, members, latitudes, longitudes).
    :param target_categorical_climatology: Climatology data for the target categories, with shape (times, ..., classes).
    :return: The Brier skill score for the classification of the target variable by the clusters.
    """
    cluster_probs_testing, cluster_probs_training = _get_pca_cluster_probabilities(
        pca_kmeans_predictor=pca_kmeans_predictor,
        input_test=input_test,
        input_climatology=input_climatology,
    )

    conditional_probs_targets = _compute_optional_conditional_probs(
        cluster_probs_training=cluster_probs_training,
        targets_categorical_climatology=targets_categorical_climatology,
    )

    return compute_BSS_clusters_target(
        cluster_probs=cluster_probs_testing,
        targets_categorical=targets_categorical_test,
        targets_categorical_climatology=targets_categorical_climatology,
        conditional_probs=conditional_probs_targets,
    )


def compute_quantile_exceedance_BSS_from_pca(
    pca_kmeans_predictor: PCAKmeansPredictor,
    input_test: np.ndarray,
    target_test: np.ndarray,
    input_climatology: np.ndarray = None,
    target_climatology: np.ndarray = None,
    q: float = 0.1,
    larger_than: bool = False,
) -> float:
    """
    Calculate the Brier skill score of test data for the classification of the exceedance of a quantile threshold by a PCA + Kmeans model (separately for each country and then aggregated). Default is below 5th percentile. The conditional probabilities of the targets given the clusters and the baseline forecast (climatology) are computed from the training data, if provided. If no training data is provided this will be computed from the test data.

    :param pca_kmeans_predictor: The PCA + Kmeans model used to predict cluster assignments
    :param input_test: Test data for the input variable, with shape (times, latitudes, longitudes)
        or, for an ensemble, (times, members, latitudes, longitudes).
    :param target_test: Test data for the target variable, with shape (times, ...) or (times, members, ...).
    :param input_climatology: Climatology data for the input variable, with shape (times, latitudes, longitudes)
        or, for an ensemble, (times, members, latitudes, longitudes).
    :param target_climatology: Climatology data for the target variable, with shape (times, ...) or (times, members, ...).
    :param q: The quantile threshold, between 0 and 1.
    :param larger_than: If True, the exceedance is defined as targets > quantile. If False, the exceedance is defined as targets < quantile.
    :return: The Brier skill score for the classification of the target variable by the clusters.
    """

    cluster_probs_testing, cluster_probs_training = _get_pca_cluster_probabilities(
        pca_kmeans_predictor=pca_kmeans_predictor,
        input_test=input_test,
        input_climatology=input_climatology,
    )

    (
        cluster_probs,
        targets_categorical,
        targets_categorical_climatology,
        conditional_probs_targets,
    ) = compute_inputs_to_calc_quantile_exceedance(
        cluster_probs_testing=cluster_probs_testing,
        cluster_probs_training=cluster_probs_training,
        targets_testing=target_test,
        targets_training=target_climatology,
        q=q,
        larger_than=larger_than,
    )

    return compute_BSS_clusters_target(
        cluster_probs=cluster_probs,
        targets_categorical=targets_categorical,
        targets_categorical_climatology=targets_categorical_climatology,
        conditional_probs=conditional_probs_targets,
    )


def compute_BSS_clusters_target_from_vae(
    vae: VAEPredictor,
    input_test: np.ndarray,
    targets_categorical_test: np.ndarray,
    input_climatology: np.ndarray = None,
    targets_categorical_climatology: np.ndarray = None,
) -> float:
    """
    Calculate the Brier skill score of test data for the classification of cluster assignments predicted by a VAE. The conditional probabilities of the targets given the clusters and the baseline forecast (climatology) are computed from the training data, if provided. If no training data is provided this will be computed from the test data.

    :param vae: The VAE model used to predict cluster assignments
    :param input_test: Test data for the input variable, with shape (times, latitudes, longitudes)
        or, for an ensemble, (times, members, latitudes, longitudes).
    :param targets_categorical_test: Test data for the target categories, with shape (times, ..., classes).
    :param input_climatology: Climatology data for the input variable, with shape (times, latitudes, longitudes)
        or, for an ensemble, (times, members, latitudes, longitudes).
    :param target_categorical_climatology: Climatology data for the target categories, with shape (times, ..., classes).
    :return: The Brier skill score for the classification of the target variable by the clusters.
    """
    cluster_probs_testing, cluster_probs_training = _get_vae_cluster_probabilities(
        vae, input_test, input_climatology,  
    )

    conditional_probs_targets = _compute_optional_conditional_probs(
        cluster_probs_training=cluster_probs_training,
        targets_categorical_climatology=targets_categorical_climatology,
    )

    return compute_BSS_clusters_target(
        cluster_probs=cluster_probs_testing,
        targets_categorical=targets_categorical_test,
        targets_categorical_climatology=targets_categorical_climatology,
        conditional_probs=conditional_probs_targets,
    )


def compute_quantile_exceedance_BSS_from_vae(
    vae: VAEPredictor,
    input_test: np.ndarray,
    target_test: np.ndarray,
    input_climatology: np.ndarray = None,
    target_climatology: np.ndarray = None,
    q: float = 0.1,
    larger_than: bool = False,
) -> float:
    """
    Calculate the Brier skill score of test data for the classification of the exceedance of a quantile threshold by VAE model (separately for each country and then aggregated). Default is below 5th percentile. The conditional probabilities of the targets given the clusters and the baseline forecast (climatology) are computed from the training data, if provided. If no training data is provided this will be computed from the test data.

    :param vae: The VAE model used to predict cluster assignments
    :param input_test: Test data for the input variable, with shape (times, latitudes, longitudes)
        or, for an ensemble, (times, members, latitudes, longitudes).
    :param target_test: Test data for the target variable, with shape (times, ...) or (times, members, ...).
    :param input_climatology: Climatology data for the input variable, with shape (times, latitudes, longitudes)
        or, for an ensemble, (times, members, latitudes, longitudes).
    :param target_climatology: Climatology data for the target variable, with shape (times, ...) or (times, members, ...).
    :param q: The quantile threshold, between 0 and 1.
    :param larger_than: If True, the exceedance is defined as targets > quantile. If False, the exceedance is defined as targets < quantile.
    :return: The Brier skill score for the classification of the target variable by the clusters.
    """

    cluster_probs_testing, cluster_probs_training = _get_vae_cluster_probabilities(
        vae,input_test,input_climatology,
    )

    (
        cluster_probs,
        targets_categorical,
        targets_categorical_climatology,
        conditional_probs_targets,
    ) = compute_inputs_to_calc_quantile_exceedance(
        cluster_probs_testing=cluster_probs_testing,
        cluster_probs_training=cluster_probs_training,
        targets_testing=target_test,
        targets_training=target_climatology,
        q=q,
        larger_than=larger_than,
    )
    return compute_BSS_clusters_target(
        cluster_probs=cluster_probs,
        targets_categorical=targets_categorical,
        targets_categorical_climatology=targets_categorical_climatology,
        conditional_probs=conditional_probs_targets,
    )

def _compute_optional_conditional_probs(
    cluster_probs_training: np.ndarray | None,
    targets_categorical_climatology: np.ndarray | None,
) -> np.ndarray | None:
    """Compute conditional probabilities if climatology data is present for both inputs."""

    if cluster_probs_training is None or targets_categorical_climatology is None:
        return None

    return compute_conditional_probabilities(
        cluster_probs=cluster_probs_training,
        targets_categorical=targets_categorical_climatology,
    )

def _get_vae_cluster_probabilities(
    vae: VAEPredictor,
    input_test: np.ndarray,
    input_climatology: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract cluster probabilities for test and optional climatology inputs using VAE.
    
    :param vae: The VAE model used to predict cluster assignments
    :param input_test: Test data for the input variable, with shape (times, latitudes, longitudes) or (times, members, latitudes, longitudes).
    :param input_climatology: Optional climatology data for the input variable, with shape (times, latitudes, longitudes) or (times, members, latitudes, longitudes).
    :return: A tuple containing:
        - cluster_probs_testing: Cluster probabilities for the test input, shape (times, n_clusters).
        - cluster_probs_training: Cluster probabilities for the climatology input, shape (times, n_clusters) or None if input_climatology is not provided.
    """
    cluster_probs_testing = predict_clusters(vae, input_test)

    if input_climatology is None:
        return cluster_probs_testing, None

    cluster_probs_training = predict_clusters(vae, input_climatology)
    return cluster_probs_testing, cluster_probs_training

def _get_pca_cluster_probabilities(
    pca_kmeans_predictor: PCAKmeansPredictor,
    input_test: np.ndarray,
    input_climatology: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract and transpose cluster probabilities for test and optional climatology inputs using PCA + Kmeans.

    :param pca_kmeans_predictor: The PCA + Kmeans model used to predict cluster assignments.
    :param input_test: Test data for the input variable.
    :param input_climatology: Optional climatology data for the input variable.
    :return: A tuple containing:
        - cluster_probs_testing: Transposed cluster probabilities for test input, shape (clusters, times).
        - cluster_probs_training: Transposed cluster probabilities for climatology input, shape (clusters, times) or None.
    """
    X_test = flatten_input(input_test)
    cluster_probs_testing = pca_kmeans_predictor.clusters(x=X_test).T

    if input_climatology is None:
        return cluster_probs_testing, None

    X_train = flatten_input(input_climatology)
    cluster_probs_training = pca_kmeans_predictor.clusters(x=X_train).T

    return cluster_probs_testing, cluster_probs_training

def predict_clusters(
    vae: VAEPredictor,
    inputs: np.ndarray,
) -> np.ndarray:
    """
    From a set of input data (e.g., Z500), calculate cluster assignment probabilities
    for each time step using a CMM-VAE.

    :param inputs: Input data to be labeled, with shape (times, latitudes, longitudes)
        or, for an ensemble, (times, members, latitudes, longitudes).
    :param vae: The CMM-VAE used to predict cluster assigments
    :return: An array of cluster assignment probabilities for each time step, with shape
        (times, vae.cfg.cluster_number)
    """
    
    if inputs.ndim == 3:
        encoder_input = flatten_input(X=inputs)
        return vae.encode(input=encoder_input)["clusters_pred"]

    else:
        n_times, n_members, n_latitudes, n_longitudes = inputs.shape
        ensemble_inputs = inputs.reshape(n_times * n_members, n_latitudes, n_longitudes)
        cluster_probabilities = vae.encode(input=flatten_input(X=ensemble_inputs))["clusters_pred"]
        return cluster_probabilities.reshape(
            n_times, n_members, vae.cfg.cluster_number
        ).mean(axis=1)






def calculate_cluster_centers(vae: VAEPredictor) -> np.ndarray:
    """
    Get cluster centers in input space by decoding the means of the mixture components
    in latent space.

    :param vae: The CMM-VAE used to calculate cluster centers
    :return: The cluster centers in input space (array of shape
        (vae.cfg.cluster_number, vae.cfg.original_dim))
    """
    mu = vae.get_mixture_components()["mu"][0]
    return vae.decode(z=mu)["x_recon"]
