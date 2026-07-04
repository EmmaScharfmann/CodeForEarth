import numpy as np
import xarray as xr
from package_name.inference.predictor import VAEPredictor
from package_name.evaluation.utils import predict_clusters


def compute_BSS_quantile_exceedance(
    vae: VAEPredictor, inputs: np.ndarray, targets: np.ndarray, q: float
) -> float:
    """
    Compute the Brier skill score for the classification of the exceedance of a quantile
    threshold by a set of CMM-VAE clusters. For example, if 'targets' is a precipitation
    field and q = 0.95, the score will measure how well the clusters predict the
    exceedance of the 95th percentile of precipitation (the quantile is computed
    separately for each grid point).

    :param vae: The CMM-VAE used to compute clusters
    :param inputs: The input data (e.g., z500) for all time steps. Shape (# times,
                   # latitudes, # longitudes)
    :param targets: The target data for all time steps. Shape (# times, ...)
    :param q: The quantile threshold, between 0 and 1.
    """
    target_quantile = np.nanquantile(targets, q=q, axis=0)
    target_quantile_exceedance = (targets > target_quantile).astype(int)

    return compute_BSS_clusters_target(vae, inputs, target_quantile_exceedance)


def compute_BSS_quantile_prediction(
    vae: VAEPredictor, inputs: np.ndarray, targets: np.ndarray, N: int
) -> float:
    """
    Compute the Brier skill score for the classification of the quantile indices of a
    target variable by a set of CMM-VAE clusters. For example, if 'targets' is a
    precipitation field and N = 4, the score will measure how well the clusters predict
    the quartile of precipitation at each grid point (the quantiles are computed
    separately for each grid point).

    :param vae: The CMM-VAE used to compute clusters
    :param inputs: The input data (e.g., z500) for all time steps. Shape (# times,
                   # latitudes, # longitudes)
    :param targets: The target data for all time steps. Shape (# times, ...)
    :param N: The number of quantiles to consider.
    """
    quantiles_one_hot = _compute_one_hot_quantiles(targets, N)

    return compute_BSS_clusters_target(vae, inputs, quantiles_one_hot)


def compute_BSS_clusters_target(
    vae: VAEPredictor, inputs: np.ndarray, targets_categorical: np.ndarray
) -> float:
    """
    Compute the Brier skill score for the classification of a binary or categorical
    target variable by a set of CMM-VAE clusters.

    :param vae: The CMM-VAE used to compute clusters
    :param inputs: The input data (e.g., z500) for all time steps. Shape (# times,
                   # latitudes, # longitudes)
    :param targets_categorical: The target data for all time steps. Must be categorical or binary
                    (e.g., exceedance of a precipitation threshold). Shape (# times, ...,
                    # classes), with the last dimension being a one-hot encoding of the
                    class. All other dimensions will be pooled together for the
                    calculation of the score.
    """
    cluster_labels = predict_clusters(vae, inputs)

    n_times = targets_categorical.shape[0]
    n_classes = targets_categorical.shape[-1]

    targets_categorical = targets_categorical.reshape(n_times, -1, n_classes)
    # Get the mean target for each cluster, and assign it to all time steps that fall in that cluster
    targets_categorical_from_clusters = _compute_target_from_cluster(
        targets_categorical, cluster_labels
    )
    # Baseline target is the mean target over all time steps, which is used as a reference for the Brier skill score
    baseline = targets_categorical.mean(axis=0)

    return _compute_brier_skill_score(
        targets_categorical,
        targets_categorical_from_clusters,
        baseline,
    )


def _compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute the Brier score for multi-class classification.

    :param y_true: True labels (one-hot encoded), with shape (n_samples, n_classes).
    :param y_prob: Predicted probabilities for each class, with shape (None, n_classes).
    :return:       Brier score.
    """

    brier_score = np.mean(np.sum((y_prob - y_true) ** 2, axis=1))
    #brier_score = np.mean(np.sum((y_prob - y_true) ** 2, axis=-1), axis=0)  # mean over samples, sum over classes

    return brier_score


def _compute_brier_skill_score(
    y_true: np.ndarray, y_prob: np.ndarray, y_prob_ref: np.ndarray
) -> float:
    """
    Compute the Brier skill score for multi-class classification.
    0 means the performance is the same as the reference model, 1 means perfect
    performance, and negative values mean worse than the reference model.

    :param y_true: True labels (one-hot encoded), with shape (n_samples, n_classes).
    :param y_prob: Predicted probabilities for each class, with shape (n_samples, n_classes).
    :param y_prob_ref: Predicted probabilities from a reference model for each class, with
                       shape (None, n_classes).
    :return:       Brier skill score.
    """

    brier_score_model = _compute_brier_score(y_true, y_prob)
    brier_score_ref = _compute_brier_score(y_true, y_prob_ref)

    return 1 - (brier_score_model / brier_score_ref)


def _compute_mean_target_per_cluster(
    targets: np.ndarray, cluster_labels: np.ndarray
) -> np.ndarray:
    """
    For an array of targets and an array of cluster labels, compute the mean
    target for each cluster. Uses xarray groupby.

    :param targets: An array of shape (n_samples, ...) containing target data (usually binary or categorical)
    :param cluster_labels: An array of shape (n_samples,) containing the cluster labels
    :return: An array of shape (n_clusters, ...) containing the mean target for each cluster
    """
    n_dims = len(targets.shape)
    targets_xr = xr.DataArray(targets, dims=[f"dim{i}" for i in range(n_dims)])
    targets_xr = targets_xr.assign_coords(cluster=("dim0", cluster_labels))
    return targets_xr.groupby("cluster").mean().values


def _compute_target_from_cluster(
    targets: np.ndarray, cluster_labels: np.ndarray
) -> np.ndarray:
    """
    For an array of targets and an array of cluster labels, compute the mean
    target for each cluster and assign it to all the samples that fall in that cluster.

    :param targets: An array of shape (n_samples, ...) containing target data (usually binary or categorical)
    :param cluster_labels: An array of shape (n_samples,) containing the cluster labels
    :return: An array of shape (n_samples, ...) containing the targets assigned to each sample based on its cluster label
    """
    mean_target_per_cluster = _compute_mean_target_per_cluster(targets, cluster_labels)
    return mean_target_per_cluster[cluster_labels.astype(np.int32)]


def _compute_quantile_indices(x: np.ndarray, N: int) -> np.ndarray:
    """
    From an array x, of shape (n0, .., np), compute the quantile indices of x along
    its first axis, for N quantiles. The output is an array of shape (n0, .., np) with
    values in {0, 1, ..., N-1}.
    For example, if N=4 and x[0,0] is in the third quartile of the distribution of
    x[:,0], then output[0,0] = 2.

    :param x: Input array for which quantiles indices will be computed
    :param N: Number of quantiles
    :return: Quantile indices of x along its first axis.
    """
    original_shape = x.shape
    n0 = original_shape[0]

    x_reshaped = x.reshape(n0, -1)

    quantiles = np.nanquantile(
        x_reshaped, q=np.linspace(0, 1, N + 1)[1:], axis=0
    )  # shape (N, n1*...*np)
    quantiles = quantiles.T  # shape (n1*...*np, N)

    quantile_indices = np.sum(
        x_reshaped[..., np.newaxis] >= quantiles[np.newaxis, :], axis=-1
    )  # shape (n0, n1*...*np)
    return quantile_indices.reshape(*original_shape)


def _compute_one_hot_quantiles(x: np.ndarray, N: int) -> np.ndarray:
    """
    From an array x, of shape (n0, .., np), compute the one-hot encoding of the
    quantile indices of x along its first axis, for N quantiles. The output is an array
    of shape (n0, .., np, N).
    For example, if N=4 and x[0,0] is in the third quartile of the distribution of
    x[:,0], then output[0,0] = [0, 0, 1, 0].

    :param x: Input array for which one-hot encoding of quantiles will be computed
    :param N: Number of quantiles
    """
    quantile_indices = _compute_quantile_indices(x, N)
    one_hot = (quantile_indices[..., None] == np.arange(N)).astype(int)
    return one_hot
