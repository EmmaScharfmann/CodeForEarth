import numpy as np
from package_name.inference.predictor import VAEPredictor
from package_name.evaluation.utils import predict_clusters
from package_name.constants import EPSILON


def compute_BSS_quantile_exceedance(
    vae: VAEPredictor, inputs: np.ndarray, targets: np.ndarray, q: float
) -> float:
    """
    Compute the Brier skill score for the classification of the exceedance of a quantile
    threshold by a set of CMM-VAE clusters.

    For example, if 'targets' is a precipitation field and q = 0.95, the score will
    measure how well the clusters predict the exceedance of the 95th percentile of
    precipitation (the quantile is computed separately for each grid point).

    :param vae: The CMM-VAE used to compute clusters
    :param inputs: The input data (e.g., z500) for all time steps. Shape (# times,
        # latitudes, # longitudes)
    :param targets: The target data for all time steps. Shape (# times, ...)
    :param q: The quantile threshold, between 0 and 1.
    :return: Brier skill score for the classification of the exceedance of the
        quantile threshold by the clusters.
    """
    target_quantile = np.nanquantile(targets, q=q, axis=0)
    exceedance = (targets > target_quantile).astype(int)
    one_hot_exceedance = np.stack([1 - exceedance, exceedance], axis=-1)

    return compute_BSS_clusters_target(
        vae=vae, inputs=inputs, targets_categorical=one_hot_exceedance
    )


def compute_BSS_quantile_prediction(
    vae: VAEPredictor, inputs: np.ndarray, targets: np.ndarray, N: int
) -> float:
    """
    Compute the Brier skill score for the classification of the quantile indices of a
    target variable by a set of CMM-VAE clusters.

    For example, if 'targets' is a precipitation field and N = 4, the score will measure
    how well the clusters predict the quartile of precipitation at each grid point (the
    quantiles are computed separately for each grid point).

    :param vae: The CMM-VAE used to compute clusters
    :param inputs: The input data (e.g., z500) for all time steps. Shape (# times,
        # latitudes, # longitudes)
    :param targets: The target data for all time steps. Shape (# times, ...)
    :param N: The number of quantiles to consider.
    :return: Brier skill score for the classification of the quantile indices of the
        target variable by the clusters.
    """
    one_hot_quantiles = _compute_one_hot_quantiles(x=targets, N=N)

    return compute_BSS_clusters_target(
        vae=vae, inputs=inputs, targets_categorical=one_hot_quantiles
    )


def compute_BSS_clusters_target(
    vae: VAEPredictor, inputs: np.ndarray, targets_categorical: np.ndarray
) -> float:
    """
    Compute the Brier skill score for the classification of a binary or categorical
    target variable by a set of CMM-VAE clusters.

    This is done probabilistically: for each time step, the predicted cluster
    probabilities are combined with the conditional probabilities of the target variable
    given each cluster to produce a forecast of the target variable. The Brier skill
    score is then computed by comparing the Brier score of the forecast to that of the
    climatological baseline.

    :param vae: The CMM-VAE used to compute clusters
    :param inputs: The input data (e.g., z500) for all time steps. Shape (# times,
        # latitudes, # longitudes)
    :param targets_categorical: The target data for all time steps. Must be categorical
        or binary (e.g., exceedance of a precipitation threshold). Shape (# times, ...,
        # classes), with the last dimension being a one-hot encoding of the class. All
        other dimensions will be pooled together for the calculation of the score.
    :return: Brier skill score for the classification of the target variable by the
             clusters.
    """
    cluster_probs = predict_clusters(vae=vae, inputs=inputs)

    n_times = targets_categorical.shape[0]
    n_classes = targets_categorical.shape[-1]

    targets_reshaped = targets_categorical.reshape(n_times, -1, n_classes)
    forecast = _compute_probabilistic_forecast(
        targets=targets_reshaped.astype(np.float32),
        cluster_probs=cluster_probs.astype(np.float32),
    )
    baseline = targets_reshaped.mean(axis=0)

    return _compute_brier_skill_score(
        y_true=targets_reshaped, y_prob=forecast, y_prob_ref=baseline
    )


def _compute_probabilistic_forecast(
    targets: np.ndarray, cluster_probs: np.ndarray
) -> np.ndarray:
    """
    Compute target probabilities across space and classes by combining the predicted
    cluster probabilities with the conditional probabilities of the target variable
    given each cluster. This uses the law of total probability, i.e.,
    P(target) = sum_c P(target | cluster=c) * P(cluster=c).

    :param targets: The target data for all time steps. Must be a one-hot encoding of
    target classes, with shape (n_times, n_spatial, n_classes).
    :param cluster_probs: The predicted cluster probabilities for all time steps, with
    shape (n_times, n_clusters).
    :return: The forecasted target probabilities for all time steps, with shape
    (n_times, n_spatial, n_classes).
    """
    n_times, n_spatial, n_classes = targets.shape
    targets_flat = targets.reshape(n_times, -1)

    conditional_probs = _compute_conditional_probabilities(
        targets_flat=targets_flat, cluster_probs=cluster_probs
    )

    # (n_times, n_clusters) @ (n_clusters, n_spatial * n_classes) -> (n_times, n_spatial * n_classes)
    forecast_flat = cluster_probs @ conditional_probs

    return forecast_flat.reshape(n_times, n_spatial, n_classes)


def _compute_conditional_probabilities(
    targets_flat: np.ndarray, cluster_probs: np.ndarray
) -> np.ndarray:
    """
    Apply Bayes' theorem to calculate the conditional target probabilities given a
    cluster assignment. This calculates the term
    P(target=t | cluster=c) = P(cluster=c and target=t)/ P(cluster=c)

    To calculate P(cluster=c and target=t), we simpy calculate the average of
    P(cluster=c) on days when target=t. This is what the first line of the function does,
    leveraging the fact that the targets are one-hot encoded.

    :param targets_flat: The target data for all time steps, flattened across all
        spatial dimensions. Must be a one-hot encoding of target classes, with shape
        (n_times, n_spatial * n_classes).
    :param cluster_probs: The predicted cluster probabilities for all time steps, with
        shape (n_times, n_clusters).
    :return: The conditional target probabilities given each cluster, with shape
        (n_clusters, n_spatial * n_classes).

    """
    joint_weights = cluster_probs.T @ targets_flat / targets_flat.shape[0]
    mean_cluster_probs = cluster_probs.mean(axis=0)[:, np.newaxis]

    return joint_weights / np.maximum(mean_cluster_probs, EPSILON)


def _compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute the Brier score for multi-class classification.

    :param y_true: True labels (one-hot encoded), with shape (..., n_classes).
    :param y_prob: Predicted probabilities for each class, with shape (..., n_classes).
    :return:       Brier score.
    """
    return np.mean(np.sum((y_prob - y_true) ** 2, axis=-1))


def _compute_brier_skill_score(
    y_true: np.ndarray, y_prob: np.ndarray, y_prob_ref: np.ndarray
) -> float:
    """
    Compute the Brier skill score for multi-class classification.
    0 means the performance is the same as the reference model, 1 means perfect
    performance, and negative values mean worse than the reference model.

    :param y_true: True labels (one-hot encoded), with shape (..., n_classes).
    :param y_prob: Predicted probabilities for each class, with shape (..., n_classes).
    :param y_prob_ref: Predicted probabilities from a reference model for each class,
        with shape (..., n_classes).
    :return: Brier skill score.
    """
    brier_score_model = _compute_brier_score(y_true=y_true, y_prob=y_prob)
    brier_score_ref = _compute_brier_score(y_true=y_true, y_prob=y_prob_ref)

    return 1.0 - (brier_score_model / brier_score_ref)


def _compute_one_hot_quantiles(x: np.ndarray, N: int) -> np.ndarray:
    """
    From an array x, of shape (n0, .., np), compute the one-hot encoding of the
    quantile indices of x along its first axis, for N quantiles. The output is an array
    of shape (n0, .., np, N).
    For example, if N=4, p=1, and x[0,0] is in the third quartile of the distribution of
    x[:,0], then output[0,0] = [0, 0, 1, 0].

    :param x: Input array for which one-hot encoding of quantiles will be computed
    :param N: Number of quantiles
    :return: One-hot encoding of the quantile indices of x along its first axis.
    """
    original_shape = x.shape
    x_reshaped = x.reshape(original_shape[0], -1)

    quantiles = np.nanquantile(
        x_reshaped, q=np.linspace(0, 1, N + 1), axis=0
    )  # shape (N, n1*...*np)

    one_hot_quantiles = np.zeros((*x_reshaped.shape, N), dtype=np.int32)

    for i in range(N):
        x_larger_than_qi = x_reshaped >= quantiles[i, :]
        x_smaller_than_qi1 = (
            x_reshaped < quantiles[i + 1, :]
            if i < N - 1
            else x_reshaped <= quantiles[i + 1, :]
        )  # We use <= for the last quantile to include the maximum value in the last bin

        one_hot_quantiles[..., i] = x_larger_than_qi & x_smaller_than_qi1

    return one_hot_quantiles.reshape(*original_shape, N)

def calculate_cluster_brier_skill_score(
    y_true_labels: np.ndarray,
    y_forecast_prob: np.ndarray,
    n_classes: int | None = None,
) -> tuple[float, float, float]:
    """
    Calculate forecast Brier score, climatological Brier score,
    and Brier skill score.

    Parameters
    ----------
    y_true_labels
        Observed integer cluster labels with shape ``(n_samples,)``.
    y_forecast_prob
        Forecast probabilities with shape
        ``(n_samples, n_classes)``.
    n_classes
        Number of clusters. Inferred from y_forecast_prob when omitted.

    Returns
    -------
    bs_forecast
        Brier score of the forecast.
    bs_climatology
        Brier score of the climatological forecast.
    bss
        Brier skill score relative to climatology.
    """
    y_true_labels = np.asarray(
        y_true_labels,
        dtype=int,
    ).reshape(-1)

    y_forecast_prob = np.asarray(
        y_forecast_prob,
        dtype=float,
    )

    if y_forecast_prob.ndim != 2:
        raise ValueError(
            "y_forecast_prob must have shape "
            "(n_samples, n_classes)."
        )

    if n_classes is None:
        n_classes = y_forecast_prob.shape[1]

    if y_forecast_prob.shape[1] != n_classes:
        raise ValueError(
            f"Expected {n_classes} probability columns, but received "
            f"{y_forecast_prob.shape[1]}."
        )

    if len(y_true_labels) != len(y_forecast_prob):
        raise ValueError(
            "y_true_labels and y_forecast_prob must contain "
            "the same number of samples."
        )

    if np.any(
        (y_true_labels < 0)
        | (y_true_labels >= n_classes)
    ):
        raise ValueError(
            f"Labels must be between 0 and {n_classes - 1}."
        )

    if not np.allclose(
        y_forecast_prob.sum(axis=1),
        1.0,
    ):
        raise ValueError(
            "Each row of y_forecast_prob must sum to one."
        )

    y_true_one_hot = np.eye(
        n_classes,
        dtype=float,
    )[y_true_labels]

    climatology_prob = (
        np.bincount(
            y_true_labels,
            minlength=n_classes,
        )
        / len(y_true_labels)
    )

    y_climatology_prob = np.broadcast_to(
        climatology_prob,
        y_forecast_prob.shape,
    )

    bs_forecast = _compute_brier_score(
        y_true=y_true_one_hot,
        y_prob=y_forecast_prob,
    )

    bs_climatology = _compute_brier_score(
        y_true=y_true_one_hot,
        y_prob=y_climatology_prob,
    )

    if np.isclose(bs_climatology, 0):
        raise ZeroDivisionError(
            "The climatological Brier score is zero, so BSS is undefined."
        )

    bss = _compute_brier_skill_score(
        y_true=y_true_one_hot,
        y_prob=y_forecast_prob,
        y_prob_ref=y_climatology_prob,
    )

    return (
        bs_forecast,
        bs_climatology,
        bss,
    )