import numpy as np
from package_name.constants import EPSILON


def compute_inputs_to_calc_quantile_exceedance(
    cluster_probs_testing: np.ndarray,
    targets_testing: np.ndarray,
    q: float,
    cluster_probs_training: np.ndarray = None,
    targets_training: np.ndarray = None,
    larger_than: bool = True,
) -> tuple:
    """
    Compute the inputs for the function compute_BSS_clusters_target needed to calculate the Brier skill score for the classification of the exceedance of a quantile
    threshold by a set of CMM-VAE clusters.

    For example, if 'targets' is a precipitation field and q = 0.95, the score will measure how well the clusters predict the exceedance of the 95th percentile of precipitation (the quantile is computed separately for each grid point).

    :param cluster_probs_testing: The probabilities of each cluster for all time steps. Shape (# times, # clusters)
    :param cluster_probs_training: The training cluster probabilities. Shape (# times, # clusters)
    :param targets_testing: The target data for all time steps. Shape (# times, ...)
    :param targets_training: The training target data. Shape (# times, ...)
    :param q: The quantile threshold, between 0 and 1.
    :param larger_than: If True, the exceedance is defined as targets > quantile. If
        False, the exceedance is defined as targets < quantile.
    :return: Inputs needed to calculate the Brier skill score for the classification of the exceedance of the quantile threshold by the clusters.
    """
    if cluster_probs_training is None or targets_training is None:
        cluster_probs_training = cluster_probs_testing
        targets_training = targets_testing

    target_quantile = np.nanquantile(targets_training, q=q, axis=0)
    exceedance_testing = (
        targets_testing > target_quantile
        if larger_than
        else targets_testing < target_quantile
    )
    exceedance_testing = exceedance_testing.astype(int)

    exceedance_training = (
        targets_training > target_quantile
        if larger_than
        else targets_training < target_quantile
    )
    exceedance_training = exceedance_training.astype(int)

    targets_categorical_testing = np.stack(
        [1 - exceedance_testing, exceedance_testing], axis=-1
    )
    targets_categorical_training = np.stack(
        [1 - exceedance_training, exceedance_training], axis=-1
    )

    conditional_probs = compute_conditional_probabilities(
        cluster_probs=cluster_probs_training,
        targets_categorical=targets_categorical_training,
    )

    return (
        cluster_probs_testing,
        targets_categorical_testing,
        targets_categorical_training,
        conditional_probs,
    )


def compute_BSS_clusters_target(
    cluster_probs: np.ndarray,
    targets_categorical: np.ndarray,
    targets_categorical_climatology: np.ndarray = None,
    conditional_probs: np.ndarray = None,
) -> float:
    """
    Compute the Brier skill score for the classification of a binary or categorical
    target variable by a set of Z500 clusters.

    This is done probabilistically: for each time step, the predicted cluster
    probabilities are combined with the conditional probabilities of the target variable
    given each cluster to produce a forecast of the target variable. The Brier skill
    score is then computed by comparing the Brier score of the forecast to that of the
    climatological baseline.

    :param cluster_probs: The predicted cluster probabilities for all time steps, with
        shape (n_times, n_clusters).
    :param targets_categorical: The target data for all time steps. Must be categorical
        or binary (e.g., exceedance of a precipitation threshold). Shape (# times, ...,
        # classes), with the last dimension being a one-hot encoding of the class. All
        other dimensions will be pooled together for the calculation of the score.
    :param targets_categorical_climatology: Optional, probabilities for training data of the
        target variable. Shape (#times, # classes,). If not provided, these will be computed
        from the targets_categorical.
    :param conditional_probs: Optional pre-computed conditional probabilities of the
        target variable given each cluster, with shape (n_clusters, n_spatial * n_classes).
        If not provided, these will be computed from the inputs and targets.
    :return: Brier skill score for the classification of the target variable by the
             clusters.
    """
    n_times = targets_categorical.shape[0]
    n_classes = targets_categorical.shape[-1]

    targets_reshaped = targets_categorical.reshape(n_times, -1, n_classes)
    targets_flat = targets_reshaped.reshape(n_times, -1)

    if targets_categorical_climatology is None:
        targets_categorical_climatology = targets_categorical

    if conditional_probs is None:
        conditional_probs = compute_conditional_probabilities(
            cluster_probs=cluster_probs, targets_categorical=targets_categorical
        )

    forecast = _compute_probabilistic_forecast(
        cluster_probs=cluster_probs.astype(np.float32),
        conditional_probs=conditional_probs.astype(np.float32),
    ).reshape(*targets_reshaped.shape)

    return compute_brier_skill_score(
        y_true=targets_reshaped,
        y_prob=forecast,
        y_prob_ref=np.mean(targets_categorical_climatology, axis=0),
    )


def compute_conditional_probabilities(
    cluster_probs: np.ndarray, targets_categorical: np.ndarray
) -> np.ndarray:
    """
    Compute the conditional probabilities of the target variable given each Z500 cluster.
    This calculates P(target=t | cluster=c) for all t and c using Bayes' theorem:
    P(target=t | cluster=c) = P(cluster=c and target=t)/ P(cluster=c).

    To calculate P(cluster=c and target=t), we simpy calculate the average of
    P(cluster=c) on days when target=t. This is what the third line of the function does,
    leveraging the fact that the targets are one-hot encoded.

    :param cluster_probs: The predicted cluster probabilities for all time steps, with
        shape (n_times, n_clusters).
    :param targets_categorical: The target data for all time steps. Must be categorical
        or binary (e.g., exceedance of a precipitation threshold). Shape (# times, ...,
        # classes), with the last dimension being a one-hot encoding of the class.
    :return: The conditional probabilities of the target variable given each cluster,
        with shape (n_clusters, n_spatial * n_classes). n_spatial is the product of the
        "..." dimensions of targets_categorical.
    """
    n_times = targets_categorical.shape[0]
    targets_flat = targets_categorical.reshape(n_times, -1)

    joint_weights = cluster_probs.T @ targets_flat / targets_flat.shape[0]
    mean_cluster_probs = cluster_probs.mean(axis=0)[:, np.newaxis]

    return joint_weights / np.maximum(mean_cluster_probs, EPSILON)


def _compute_probabilistic_forecast(
    cluster_probs: np.ndarray,
    conditional_probs: np.ndarray,
) -> np.ndarray:
    """
    Compute target probabilities across space and classes by combining the predicted
    cluster probabilities with the conditional probabilities of the target variable
    given each cluster. This uses the law of total probability, i.e.,
    P(target) = sum_c P(target | cluster=c) * P(cluster=c).

    :param cluster_probs: The predicted cluster probabilities for all time steps, with
    shape (n_times, n_clusters).
    :param conditional_probs: The conditional probabilities of the target variable
        given each cluster, with shape (n_clusters, n_spatial * n_classes).
    :return: The forecasted target probabilities for all time steps, with shape
    (n_times, n_spatial * n_classes).
    """
    # (n_times, n_clusters) @ (n_clusters, n_spatial * n_classes) -> (n_times, n_spatial * n_classes)
    forecast_flat = cluster_probs @ conditional_probs

    return forecast_flat


def _compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute the Brier score for multi-class classification.

    :param y_true: True labels (one-hot encoded), with shape (..., n_classes).
    :param y_prob: Predicted probabilities for each class, with shape (..., n_classes).
    :return:       Brier score.
    """
    return np.mean(np.sum((y_prob - y_true) ** 2, axis=-1))


def compute_brier_skill_score(
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


def calculate_forecast_brier_skill_score(
    y_true_labels: np.ndarray,
    y_forecast_prob: np.ndarray,
    n_classes: int | None = None,
) -> tuple[float, float, float]:
    """
    Calculate forecast Brier score, climatological Brier score,
    and Brier skill score.

    y_true_labels: Observed integer cluster labels with shape (n_samples,).
    y_forecast_prob: Forecast probabilities with shape (n_samples, n_classes).
    n_classes: Number of clusters. Inferred from y_forecast_prob when omitted.

    returns: Tuple of (bs_forecast, bs_climatology, bss):
    bs_forecast: Brier score of the forecast.
    bs_climatology: Brier score of the climatological forecast.
    bss: Brier skill score relative to climatology.
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
        raise ValueError("y_forecast_prob must have shape " "(n_samples, n_classes).")

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

    if np.any((y_true_labels < 0) | (y_true_labels >= n_classes)):
        raise ValueError(f"Labels must be between 0 and {n_classes - 1}.")

    if not np.allclose(
        y_forecast_prob.sum(axis=1),
        1.0,
    ):
        raise ValueError("Each row of y_forecast_prob must sum to one.")

    y_true_one_hot = np.eye(
        n_classes,
        dtype=float,
    )[y_true_labels]

    if len(y_true_labels) == 0:
        raise ValueError(
            "y_true_labels and y_forecast_prob must contain " "at least one sample."
        )
    climatology_prob = np.bincount(
        y_true_labels,
        minlength=n_classes,
    ) / len(y_true_labels)

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
