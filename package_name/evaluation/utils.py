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
    X_test = flatten_input(input_test)
    cluster_probs_testing = pca_kmeans_predictor.clusters(x=X_test)

    if input_climatology is None or targets_categorical_climatology is None:
        conditional_probs_targets = None

    else:
        X_train = flatten_input(input_climatology)
        cluster_probs_training = pca_kmeans_predictor.clusters(x=X_train)
        conditional_probs_targets = compute_conditional_probabilities(
            cluster_probs=cluster_probs_training.T,
            targets_categorical=targets_categorical_climatology,
        )

    return compute_BSS_clusters_target(
        cluster_probs=cluster_probs_testing.transpose(1, 0),
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

    X_test = flatten_input(input_test)
    cluster_probs_testing = pca_kmeans_predictor.clusters(x=X_test)

    if input_climatology is None or target_climatology is None:
        cluster_probs_training = None
        conditional_probs_targets = None
    else:
        X_train = flatten_input(input_climatology)
        cluster_probs_training = pca_kmeans_predictor.clusters(x=X_train)

    (
        cluster_probs,
        targets_categorical,
        targets_categorical_climatology,
        conditional_probs_targets,
    ) = compute_inputs_to_calc_quantile_exceedance(
        cluster_probs_testing=cluster_probs_testing.transpose(1, 0),
        cluster_probs_training=cluster_probs_training.transpose(1, 0),
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

    cluster_probs_testing = predict_clusters(vae, input_test)

    if input_climatology is None or targets_categorical_climatology is None:
        conditional_probs_targets = None

    else:
        cluster_probs_training = predict_clusters(vae, input_climatology)
        conditional_probs_targets = compute_conditional_probabilities(
            cluster_probs=cluster_probs_training,
            targets_categorical=targets_categorical_climatology,
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

    cluster_probs_testing = predict_clusters(vae, input_test)

    if input_climatology is None or target_climatology is None:
        cluster_probs_training = None
        conditional_probs_targets = None
    else:
        cluster_probs_training = predict_clusters(vae, input_climatology)

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

    if inputs.ndim == 4:
        n_times, n_members, n_latitudes, n_longitudes = inputs.shape
        ensemble_input = inputs.reshape(n_times * n_members, n_latitudes, n_longitudes)
        cluster_probabilities = vae.encode(input=flatten_input(X=ensemble_input))[
            "clusters_pred"
        ]
        return cluster_probabilities.reshape(
            n_times, n_members, vae.cfg.cluster_number
        ).mean(axis=1)

    raise ValueError(
        "inputs must have shape (times, latitudes, longitudes) or "
        "(times, members, latitudes, longitudes)"
    )


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
