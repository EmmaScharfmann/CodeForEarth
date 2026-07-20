import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
from package_name.evaluation.metrics import calculate_cluster_brier_skill_score
from typing import Sequence
from pathlib import Path

METHOD_NAMES = {
    "cmmvae": "CMM-VAE",
    "pca": "PCA",
}
def plot_forecast_scores(
    methods: Sequence[str],
    cluster_number: int,
    chosen_count: int = 1,
    n_bootstrap: int = 100,
    sample_fraction: float = 0.9,
    random_state: int | None = None,
    save_results: bool = True,
):
    """
    Calculate and plot BSS and ROC AUC for multiple methods.

    BSS is bootstrapped. ROC AUC is calculated once per lead time.
    """
    bss_results = []
    roc_results = []

    if save_results:
        Path("results").mkdir(
            parents=True,
            exist_ok=True,
        )

    for method in methods:
        if method not in METHOD_NAMES:
            raise ValueError(
                f"Unknown method {method!r}. "
                f"Choose from {list(METHOD_NAMES)}."
            )

        method_name = METHOD_NAMES[method]

        print(f"Processing {method_name}")

        merged_forecast = _load_merged_forecast(
            method=method,
            cluster_number=cluster_number,
            chosen_count=chosen_count,
        )

        method_bss = bootstrap_brier_skill_score(
            merged_forecast=merged_forecast,
            method_name=method_name,
            cluster_number=cluster_number,
            n_bootstrap=n_bootstrap,
            sample_fraction=sample_fraction,
            random_state=random_state,
        )

        method_roc = roc_auc_by_leadtime(
            merged_forecast=merged_forecast,
            method_name=method_name,
            cluster_number=cluster_number,
        )

        bss_results.append(method_bss)
        roc_results.append(method_roc)

        if save_results:
            method_bss.to_csv(
                f"results/bss_s2s_{method}_bootstrapped_"
                f"{cluster_number}.csv",
                index=False,
            )

            method_roc.to_csv(
                f"results/roc_s2s_{method}_"
                f"{cluster_number}.csv",
                index=False,
            )

    combined_bss = pd.concat(
        bss_results,
        ignore_index=True,
    )

    combined_roc = pd.concat(
        roc_results,
        ignore_index=True,
    )

    return _plot_bss_and_roc(
        bss_data=combined_bss,
        roc_data=combined_roc,
    )

def _load_merged_forecast(
    method: str,
    cluster_number: int,
    chosen_count: int = 1,
) -> pd.DataFrame:
    """Load and merge forecast and ERA5 probabilities."""

    #if method is not cmmvae or pca or cca, raise error
    if method not in ["cmmvae", "pca", "cca"]:
        raise ValueError(
            f"Invalid method: {method}. Must be 'cmmvae', 'pca', or 'cca'."
        )
    
    era5_file=f"results/cond_prob_{method}_era5_{cluster_number}.csv"
    if method == "cmmvae":
        forecast_file=f"results/cond_prob_{method}_forecast_{cluster_number}_{chosen_count}.csv"
    else:
        forecast_file=f"results/cond_prob_{method}_forecast_{cluster_number}.csv"

    probability_columns = [str(i) for i in range(cluster_number)]

    era5 = pd.read_csv(era5_file)

    era5["label"] = (
        era5[probability_columns]
        .idxmax(axis=1)
        .astype(int)
    )
    era5["prob"] = era5[probability_columns].max(axis=1)

    era5 = era5.rename(
        columns={
            str(i): f"era5_{i}"
            for i in range(cluster_number)
        }
    )

    forecast = pd.read_csv(forecast_file)

    return (
        forecast
        .merge(era5, on="valid_date", how="inner")
        .dropna()
        .reset_index(drop=True)
    )
    


def bootstrap_brier_skill_score(
    merged_forecast: pd.DataFrame,
    method_name: str,
    cluster_number: int,
    n_bootstrap: int = 100,
    sample_fraction: float = 0.9,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Bootstrap Brier skill scores for every lead time."""
    rng = np.random.default_rng(random_state)

    probability_columns = [
        str(i)
        for i in range(cluster_number)
    ]

    results = []

    for leadtime, leadtime_data in merged_forecast.groupby(
        "leadtime"
    ):
        leadtime_data = leadtime_data.reset_index(drop=True)

        sample_size = max(
            1,
            round(len(leadtime_data) * sample_fraction),
        )

        for bootstrap in range(n_bootstrap):
            sample_indices = rng.integers(
                0,
                len(leadtime_data),
                size=sample_size,
            )

            sampled = leadtime_data.iloc[sample_indices]

            _, _, bss = calculate_cluster_brier_skill_score(
                y_true_labels=sampled["label"].to_numpy(),
                y_forecast_prob=sampled[
                    probability_columns
                ].to_numpy(),
                n_classes=cluster_number,
            )

            results.append(
                {
                    "Method": method_name,
                    "Leadtime": leadtime,
                    "Bootstrap": bootstrap,
                    "BSS": bss,
                }
            )

    return pd.DataFrame(results)



def roc_auc_by_leadtime(
    merged_forecast: pd.DataFrame,
    method_name: str,
    cluster_number: int,
    n_bootstrap: int = 100,
    sample_fraction: float = 0.9,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Bootstrap multiclass ROC AUC for every lead time."""
    rng = np.random.default_rng(random_state)

    probability_columns = [
        str(i)
        for i in range(cluster_number)
    ]
    expected_classes = set(range(cluster_number))

    results = []

    for leadtime, leadtime_data in merged_forecast.groupby(
        "leadtime"
    ):
        leadtime_data = leadtime_data.reset_index(drop=True)

        present_classes = set(
            leadtime_data["label"].astype(int).unique()
        )

        if present_classes != expected_classes:
            raise ValueError(
                f"Lead time {leadtime} is missing classes "
                f"{sorted(expected_classes - present_classes)}."
            )

        sample_size = max(
            1,
            round(len(leadtime_data) * sample_fraction),
        )

        for bootstrap in range(n_bootstrap):
            # Resample until every class is represented.
            for _ in range(1000):
                sample_indices = rng.integers(
                    0,
                    len(leadtime_data),
                    size=sample_size,
                )

                sampled = leadtime_data.iloc[
                    sample_indices
                ]

                truth = sampled[
                    "label"
                ].to_numpy(dtype=int)

                if set(np.unique(truth)) == expected_classes:
                    break
            else:
                raise RuntimeError(
                    f"Could not generate a valid ROC bootstrap "
                    f"sample for lead time {leadtime}."
                )

            prediction = sampled[
                probability_columns
            ].to_numpy(dtype=float)

            score = roc_auc_score(
                truth,
                prediction,
                labels=np.arange(cluster_number),
                multi_class="ovo",
                average="macro",
            )

            results.append(
                {
                    "Method": method_name,
                    "Leadtime": leadtime,
                    "Bootstrap": bootstrap,
                    "ROC_AUC": score,
                }
            )
    return pd.DataFrame(results)

def _plot_bss_and_roc(
    bss_data: pd.DataFrame,
    roc_data: pd.DataFrame,
    minimum_leadtime: int = 3,
    maximum_leadtime: int = 35,
):
    """Plot bootstrapped BSS and non-bootstrapped ROC AUC."""
    bss_plot_data = bss_data.loc[
        bss_data["Leadtime"] >= minimum_leadtime
    ]

    roc_plot_data = roc_data.loc[
        roc_data["Leadtime"] >= minimum_leadtime
    ]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13, 5),
        dpi=300,
        sharex=True,
    )

    # Brier skill score
    sns.lineplot(
        data=bss_plot_data,
        x="Leadtime",
        y="BSS",
        hue="Method",
        errorbar=("pi", 95),
        ax=axes[0],
    )

    axes[0].axhline(
        y=0,
        color="black",
        linewidth=1,
    )
    axes[0].set_title("(a) Brier skill score")
    axes[0].set_xlabel("Lead time")
    axes[0].set_ylabel("Brier skill score")
    axes[0].set_xlim(
        minimum_leadtime,
        maximum_leadtime,
    )

    # ROC AUC
    sns.lineplot(
        data=roc_plot_data,
        x="Leadtime",
        y="ROC_AUC",
        hue="Method",
        errorbar=("pi", 95),
        ax=axes[1],
    )

    axes[1].axhline(
        y=0.5,
        color="black",
        linewidth=1,
    )
    axes[1].set_title("(b) ROC AUC")
    axes[1].set_xlabel("Lead time")
    axes[1].set_ylabel("Multiclass ROC AUC")
    axes[1].set_xlim(
        minimum_leadtime,
        maximum_leadtime,
    )

    fig.tight_layout()

    return fig, axes