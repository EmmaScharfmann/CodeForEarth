import numpy as np
import pandas as pd
import seaborn as sns
from package_name.evaluation.metrics import compute_brier_skill_score, compute_probabilistic_forecast, compute_conditional_probabilities

def compute_clusters(data, model, model_type):
    """Predict clusters probabilities for given either a vae or pca model."""
    if model_type=='vae':
        encoded = model.encode(input=data)["clusters_pred"]
    elif model_type=='pca':
        encoded = model.clusters(x=data).T
    return encoded
def predict_all_hindcast_cluster_probs(z500_days_anom, model, cluster_number, model_type):
    """Reshape hindcast anomalies, encode via VAE, and average over ensemble members."""
    nlead, nmembers, ntimes, nlat, nlon = z500_days_anom.shape
    reshaped = z500_days_anom.values.reshape(
        nlead * nmembers * ntimes, nlat * nlon
    )
    encoded = compute_clusters(data=reshaped, model=model, model_type=model_type)

    return encoded.reshape(nlead, nmembers, ntimes, cluster_number)

def predict_mean_hindcast_cluster_probs(z500_days_anom, model, cluster_number, model_type):
    """Reshape hindcast anomalies, encode via VAE, and average over ensemble members."""
    cluster_probs = predict_all_hindcast_cluster_probs(z500_days_anom, model, cluster_number, model_type)
    return cluster_probs.mean(axis=1)  # Average over ensemble members

def extract_matching_era5_probs(z500_days_anom, z500, model, cluster_number, model_type):
    """Extract and encode corresponding ERA5 truth matching forecast lead times."""
    nlead, _, ntimes, nlat, nlon = z500_days_anom.shape
    era5_matched = np.zeros((nlead, ntimes, nlat, nlon), dtype=float)

    for i in range(nlead):
        for j in range(ntimes):
            forecast_time = z500_days_anom.time.values[j] + np.timedelta64(
                i, "D"
            )
            era5_matched[i, j] = z500.sel(
                time=forecast_time, method="nearest"
            ).values

    reshaped = era5_matched.reshape(nlead * ntimes, nlat * nlon)
    encoded = compute_clusters(data=reshaped, model=model, model_type=model_type)
    return encoded.reshape(nlead, ntimes, cluster_number)


def extract_matching_target_variable(z500_days_anom, targets_categorical):
    """Extract the target variable corresponding to the forecast lead times."""
    nlead, _, ntimes, nlat, nlon = z500_days_anom.shape
    _, n_target_classes = targets_categorical.shape
    target_matched = np.zeros((nlead, ntimes, n_target_classes), dtype=float)

    for i in range(nlead):
        for j in range(ntimes):
            forecast_time = z500_days_anom.time.values[j] + np.timedelta64(
                i, "D"
            )
            target_matched[i, j] = targets_categorical.sel(
                time=forecast_time, method="nearest"
            ).values

    return target_matched

def forecast_target_variable_from_hindcast(z500_days_anom, z500, model, targets_categorical, cluster_number, model_type):
    """Make a forecast of the target variable from hindcast data and compute Brier Skill Score."""
    cluster_probs = predict_all_hindcast_cluster_probs(
        z500_days_anom, 
        model, 
        cluster_number,
        model_type
        )
    nlead, nmembers, ntimes, _ = cluster_probs.shape
    cluster_probs_reshaped = cluster_probs.reshape(nlead*nmembers*ntimes, -1)

    ntimes_z500, nlat, nlon = z500.shape
    z500_reshaped = z500.values.reshape(
        ntimes_z500, nlat * nlon
    )
    cluster_probs_era5 = compute_clusters(data=z500_reshaped, model=model, model_type=model_type)
    
    conditional_probs = compute_conditional_probabilities(
        cluster_probs=cluster_probs_era5, targets_categorical=targets_categorical
    )

    forecast = compute_probabilistic_forecast(
        cluster_probs=cluster_probs_reshaped.astype(np.float32),
        conditional_probs=conditional_probs.astype(np.float32),
    )

    return forecast.reshape(nlead, nmembers, ntimes, -1).mean(axis=1)  # Reshape to (nlead, nmembers, ntimes, n_targets)


def compute_bootstrap_brier_skill_scores(
    cluster_probs,
    cluster_probs_reference,
    baseline_probs,
    n_bootstrap=100,
    subsample_ratio=0.9,
):
    """Compute bootstrapped Brier Skill Scores across lead times."""
    nlead, ntimes, _ = cluster_probs.shape
    bss_matrix = np.zeros((nlead, n_bootstrap))
    sample_size = int(ntimes * subsample_ratio)

    for j in range(nlead):
        if n_bootstrap == 1:
            bss_matrix[j, 0] = compute_brier_skill_score(
                y_true=cluster_probs_reference[j],
                y_prob=cluster_probs[j],
                y_prob_ref=baseline_probs,
            )
        else:
            for i in range(n_bootstrap):
                idx = np.random.choice(ntimes, size=sample_size, replace=True)
                bss_matrix[j, i] = compute_brier_skill_score(
                    y_true=np.take(cluster_probs_reference[j], idx, axis=0),
                    y_prob=np.take(cluster_probs[j], idx, axis=0),
                    y_prob_ref=baseline_probs,
                )
    return bss_matrix


def plot_brier_skill_scores_bootstrap(ax, bss_matrix, max_leadtime=35):
    """Plot Brier Skill Score against lead time with 95% percentile intervals."""
    df = pd.DataFrame(bss_matrix.T).melt(
        var_name="LeadTime", value_name="Brier Skill Score"
    )
    sns.lineplot(
        data=df,
        x="LeadTime",
        y="Brier Skill Score",
        errorbar=("pi", 95),
        ax=ax,
    )
    ax.set_xlim(0, max_leadtime)
    ax.axhline(0, color="k", linestyle="--")
    return ax

def plot_brier_skill_score(ax, bss, max_leadtime=35):
    """Plot Brier Skill Score against lead time with 95% percentile intervals."""
    ax.plot(np.arange(len(bss)), bss, label="Brier Skill Score")
    ax.set_xlim(0, max_leadtime)
    ax.axhline(0, color="k", linestyle="--")
    return ax