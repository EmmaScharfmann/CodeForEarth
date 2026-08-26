import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
from package_name.evaluation import metrics
from typing import Sequence
from pathlib import Path
from enum import Enum
from typing import Literal
from collections.abc import Iterator

type MethodName = Literal["cmmvae", "pca"]

_FILE_NAME = "cond_prob_{method}_{data}_{cluster_number}_{chosen_count}.csv"


def save_cprobs(
    cprobs: tuple[pd.DataFrame, ...],
    methods: Sequence[MethodName],
    data: tuple[str, ...],
    cluster_number: int,
    chosen_count: int,
    output_dir: str | Path = "results",
) -> str:
    """
    Save conditional probabilities to CSV files.
    cprobs: Tuple of DataFrames containing conditional probabilities for each method.
    methods: Sequence of method names corresponding to the cprobs.
    data: Tuple of strings indicating the type of data (e.g., "era5", "forecast").
    cluster_number: Number of clusters used in the forecast.
    chosen_count: Number of chosen samples for the CMM-VAE method. Default is 1.
    output_dir: Directory where the CSV files will be saved. Default is "results".

    Returns: Saves the conditional probabilities to CSV files in the specified output directory
            and returns a message indicating the location of the saved files.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for method, cprob, data_type in zip(methods, cprobs, data):
        file_name = _FILE_NAME.format(
            method=method,
            data=data_type,
            cluster_number=cluster_number,
            chosen_count=chosen_count if method == "cmmvae" else "",
        )
        file_path = output_dir / file_name
        cprob.to_csv(file_path, index=False)
    return f"Conditional probabilities saved to {output_dir.resolve()}"


def plot_forecast_scores(
    methods: Sequence[MethodName],
    cluster_number: int,
    chosen_count: int = 1,
    n_bootstrap: int = 100,
    sample_fraction: float = 0.9,
    result_dir: str | Path = "results",
    random_state: int | None = None,
    save_results: bool = True,
):
    """
    Calculate and plot BSS and ROC AUC for multiple methods.

    methods: List of method names to evaluate. Must be a subset of METHOD_NAMES keys.
    cluster_number: Number of clusters used in the forecast.
    chosen_count: Number of chosen samples for the CMM-VAE method. Default is 1.
    n_bootstrap: Number of bootstrap samples for BSS and ROC AUC.
    sample_fraction: Fraction of data to sample for each bootstrap iteration.
    result_dir: Directory to where conditional probabilities are saved and where BSS and ROC AUC results will be saved.
                Default is "results".
    random_state: Random seed for reproducibility.
    save_results: If True, save the BSS and ROC AUC results to CSV files in the "results" directory.

    Returns:figures of BSS and ROC AUC plots.
    """
    bss_results: list[pd.DataFrame] = []
    roc_results: list[pd.DataFrame] = []

    for method in methods:
        method_name = method.upper()

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

        if save_results is True:
            Path(result_dir).mkdir(
                parents=True,
                exist_ok=True,
            )

            method_bss.to_csv(
                f"{result_dir}/bss_s2s_{method}_bootstrapped_" f"{cluster_number}.csv",
                index=False,
            )

            method_roc.to_csv(
                f"{result_dir}/roc_s2s_{method}_" f"{cluster_number}.csv",
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
    method: MethodName,
    cluster_number: int,
    chosen_count: int = 1,
    result_dir: str | Path = "results",
) -> pd.DataFrame:
    """Load and merge forecast and ERA5 probabilities."""

    count = chosen_count if method == "cmmvae" else ""
    era5_file = _FILE_NAME.format(
        method=method, data="era5", cluster_number=cluster_number, chosen_count=count
    )
    forecast_file = _FILE_NAME.format(
        method=method,
        data="forecast",
        cluster_number=cluster_number,
        chosen_count=count,
    )

    result_dir = Path(result_dir)

    era5 = pd.read_csv(result_dir / era5_file)
    forecast = pd.read_csv(result_dir / forecast_file)

    probability_columns = [str(i) for i in range(cluster_number)]
    probabilities = era5[probability_columns]

    era5 = era5.assign(
        label=probabilities.idxmax(axis=1).astype(int),
        prob=probabilities.max(axis=1),
    ).rename(columns={column: f"era5_{column}" for column in probability_columns})

    return (
        forecast.merge(era5, on="valid_date", how="inner")
        .dropna()
        .reset_index(drop=True)
    )


def _bootstrap_samples_by_leadtime(
    merged_forecast: pd.DataFrame,
    cluster_number: int,
    n_bootstrap: int,
    sample_fraction: float,
    random_state: int | None,
    require_all_classes: bool = False,
    max_attempts: int = 1000,
) -> Iterator[tuple[int, int, pd.DataFrame]]:
    """Yield bootstrap samples for every forecast lead time."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")

    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be greater than 0 and at most 1.")

    rng = np.random.default_rng(random_state)
    expected_classes = set(range(cluster_number))

    for leadtime, leadtime_data in merged_forecast.groupby("leadtime"):
        if require_all_classes:
            present_classes = set(leadtime_data["label"].astype(int).unique())
            missing_classes = expected_classes - present_classes

            if missing_classes:
                raise ValueError(
                    f"Lead time {leadtime} is missing classes "
                    f"{sorted(missing_classes)}."
                )

        sample_size = max(
            1,
            round(len(leadtime_data) * sample_fraction),
        )

        for bootstrap in range(n_bootstrap):
            attempts = max_attempts if require_all_classes else 1

            for _ in range(attempts):
                sample_indices = rng.integers(
                    0,
                    len(leadtime_data),
                    size=sample_size,
                )
                sampled = leadtime_data.iloc[sample_indices]

                if not require_all_classes:
                    break

                sampled_classes = set(sampled["label"].astype(int).unique())
                if sampled_classes == expected_classes:
                    break
            else:
                raise RuntimeError(
                    "Could not generate a bootstrap sample containing "
                    f"every class for lead time {leadtime} after "
                    f"{max_attempts} attempts."
                )

            yield int(leadtime), bootstrap, sampled


def bootstrap_brier_skill_score(
    merged_forecast: pd.DataFrame,
    method_name: str,
    cluster_number: int,
    n_bootstrap: int = 100,
    sample_fraction: float = 0.9,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Calculate bootstrapped Brier skill score by lead time.
    merged_forecast: DataFrame containing merged forecast and ERA5 probabilities.
    method_name: Name of the method used for the forecast.
    cluster_number: Number of clusters used in the forecast.
    n_bootstrap: Number of bootstrap samples to generate for each lead time.
    sample_fraction: Fraction of data to sample for each bootstrap iteration.
    random_state: Random seed for reproducibility.

    Returns: DataFrame containing Brier skill scores for each lead time and bootstrap sample.
    """
    probability_columns = [str(cluster) for cluster in range(cluster_number)]
    results: list[dict[str, str | int | float]] = []

    samples = _bootstrap_samples_by_leadtime(
        merged_forecast=merged_forecast,
        cluster_number=cluster_number,
        n_bootstrap=n_bootstrap,
        sample_fraction=sample_fraction,
        random_state=random_state,
    )

    for leadtime, bootstrap, sampled in samples:
        _, _, bss = metrics.calculate_forecast_brier_skill_score(
            y_true_labels=sampled["label"].to_numpy(),
            y_forecast_prob=sampled[probability_columns].to_numpy(),
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

    return pd.DataFrame.from_records(results)


def roc_auc_by_leadtime(
    merged_forecast: pd.DataFrame,
    method_name: str,
    cluster_number: int,
    n_bootstrap: int = 100,
    sample_fraction: float = 0.9,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Calculate bootstrapped multiclass ROC AUC by lead time.
    merged_forecast: DataFrame containing merged forecast and ERA5 probabilities.
    method_name: Name of the method used for the forecast.
    cluster_number: Number of clusters used in the forecast.
    n_bootstrap: Number of bootstrap samples to generate for each lead time.
    sample_fraction: Fraction of data to sample for each bootstrap iteration.
    random_state: Random seed for reproducibility.

    Returns: DataFrame containing ROC AUC scores for each lead time and bootstrap sample.
    """
    probability_columns = [str(cluster) for cluster in range(cluster_number)]
    results: list[dict[str, str | int | float]] = []

    samples = _bootstrap_samples_by_leadtime(
        merged_forecast=merged_forecast,
        cluster_number=cluster_number,
        n_bootstrap=n_bootstrap,
        sample_fraction=sample_fraction,
        random_state=random_state,
        require_all_classes=True,
    )

    for leadtime, bootstrap, sampled in samples:
        truth = sampled["label"].to_numpy(dtype=int)
        prediction = sampled[probability_columns].to_numpy(dtype=float)

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

    return pd.DataFrame.from_records(results)


def _plot_bss_and_roc(
    bss_data: pd.DataFrame,
    roc_data: pd.DataFrame,
    minimum_leadtime: int = 3,
    maximum_leadtime: int = 35,
):
    """Plot bootstrapped BSS and ROC AUC.
    bss_data: DataFrame containing Brier skill scores for each lead time and bootstrap sample.
    roc_data: DataFrame containing ROC AUC scores for each lead time and bootstrap sample.
    minimum_leadtime: Minimum lead time to display on the x-axis.
    maximum_leadtime: Maximum lead time to display on the x-axis.

    Returns: Matplotlib figure and axes objects containing the BSS and ROC AUC plots."""
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13, 5),
        dpi=300,
        sharex=True,
    )

    plot_specs = (
        {
            "axis": axes[0],
            "data": bss_data,
            "metric": "BSS",
            "reference": 0.0,
            "title": "(a) Brier skill score",
            "ylabel": "Brier skill score",
        },
        {
            "axis": axes[1],
            "data": roc_data,
            "metric": "ROC_AUC",
            "reference": 0.5,
            "title": "(b) ROC AUC",
            "ylabel": "Multiclass ROC AUC",
        },
    )

    for spec in plot_specs:
        axis = spec["axis"]
        plot_data = spec["data"].loc[
            spec["data"]["Leadtime"].between(
                minimum_leadtime,
                maximum_leadtime,
            )
        ]

        sns.lineplot(
            data=plot_data,
            x="Leadtime",
            y=spec["metric"],
            hue="Method",
            errorbar=("pi", 95),
            ax=axis,
        )

        axis.axhline(
            y=spec["reference"],
            color="black",
            linewidth=1,
        )
        axis.set(
            title=spec["title"],
            xlabel="Lead time",
            ylabel=spec["ylabel"],
            xlim=(minimum_leadtime, maximum_leadtime),
        )

    fig.tight_layout()
    return fig, axes
