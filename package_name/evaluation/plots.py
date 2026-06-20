import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
from matplotlib.contour import ContourSet

from package_name.training.vae import VAE
from package_name.evaluation.utils import calculate_cluster_centers, get_input_labels
from package_name.data_processing.data_processor import unflatten_input


def plot_single_map(
    ax: plt.Axes, data: xr.DataArray, title: str, borders: bool = True, **kwargs
) -> ContourSet:
    """
    Plot data on a map

    :param ax:  The matplotlib axis on which data is plotted. Has to have been defined using a cartopy projection
    :param data: The data to be plotted, must have dimensions "latitude" and "longitude
    :param title: plot title
    :param borders: If True, add country borders to the map.
    :param kwargs:  Other arguments, passed to the contourf function for plotting.
    """
    cf = data.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), **kwargs)
    ax.coastlines()
    if borders == True:
        ax.add_feature(cfeature.BORDERS)
    ax.set_title(title)
    return cf


def plot_set_of_maps(
    data: xr.DataArray,
    titles: list[str],
    suptitle=str,
    plot_reordering: np.ndarray | None = None,
    borders: bool = True,
    projection: ccrs.Projection = ccrs.Orthographic(0, 45),
    **kwargs,
) -> plt.Figure:
    """
    Plot a list of maps

    :param data: The data to be plotted. Must have dimensions dim0, 'latitude', and 
                 'longitude'. (dim0 can have any name, but it has to be the first 
                 dimension of the data array)
    :param titles: A list of titles for each subplot.
    :param suptitle: General plot title
    :param plot_reordering: A list of indices to reorder the plots. The first data 
                            instance to be plotted will be data[plot_reordering[0]], the 
                            second cluster will be data[plot_reordering[1]], and so on. 
                            If None, data will be plotted in its original order.
    :param borders: If True, add country borders to the map.
    :param projection: The cartopy projection to use for the map.
    :param kwargs:  Other arguments, passed to the contourf function for plotting.
    """

    plot_number = data.values.shape[0]

    fig = plt.figure(figsize=(4 * plot_number, 4))
    gs = GridSpec(1, plot_number + 1, width_ratios=[1] * plot_number + [0.05])
    axs = [fig.add_subplot(gs[i], projection=projection) for i in range(plot_number)]
    cax = fig.add_subplot(gs[-1])

    if plot_reordering is None:
        plot_reordering = np.arange(plot_number)

    for i, ax in enumerate(axs):
        cf = plot_single_map(
            ax=ax,
            data=data[plot_reordering[i]],
            title=titles[plot_reordering[i]],
            borders=borders,
            add_colorbar=False,
            **kwargs,
        )

    fig.colorbar(cf, cax=cax, orientation="vertical")

    fig.suptitle(suptitle, fontsize=16)
    fig.tight_layout()


def plot_cluster_centers(
    vae: VAE,
    input_sample: xr.DataArray,
    plot_reordering: np.ndarray | None = None,
    **kwargs,
):
    """
    Plot the center of the CMM-VAE clusters in input space.

    :param vae: The CMM-VAE used to calculate cluster centers
    :param input_sample: A sample of input data (e.g., Z500) used for latitude and longitude axes
    :param plot_reordering: A list of indices to reorder the clusters for plotting.
    :param kwargs:          Other arguments, passed to the contourf function for plotting.
    """
    ny, nx = len(input_sample.latitude), len(input_sample.longitude)
    cluster_centers = calculate_cluster_centers(vae)

    cluster_centers = unflatten_input(cluster_centers, ny, nx)
    cluster_centers = xr.DataArray(
        cluster_centers,
        coords=[
            np.arange(vae.cfg.cluster_number),
            input_sample.latitude,
            input_sample.longitude,
        ],
        dims=["label", "latitude", "longitude"],
    )
    titles = [f"Cluster {i}" for i in range(vae.cfg.cluster_number)]
    plot_set_of_maps(
        cluster_centers,
        titles,
        "True cluster centers (calculated by decoding mixture components)",
        plot_reordering=plot_reordering,
        **kwargs,
    )


def plot_empirical_cluster_centers(
    input_with_labels: xr.DataArray, plot_reordering: np.ndarray | None = None, **kwargs
):
    """
    Plot cluster centers on a set of maps.

    :param input_with_labels: The input data (e.g., z500) for all time steps. Must have 
                              dimensions "time", "latitude", and "longitude". The time 
                              dimension must have a coordinate named "label" that contains 
                              the cluster labels for each time step.
    :param plot_reordering: A list of indices to reorder the clusters for plotting.
    :param kwargs:          Other arguments, passed to the contourf function for plotting.
    """

    cluster_centers = input_with_labels.groupby("label").mean()
    labels_data = input_with_labels.label.values

    cluster_number = cluster_centers.values.shape[0]
    cluster_counts = [
        labels_data[labels_data == i].shape[0] for i in range(cluster_number)
    ]
    cluster_frequencies = np.array(cluster_counts) / len(labels_data)

    titles = [
        f"Cluster {i}, {cluster_frequencies[i]*100:.1f}%" for i in range(cluster_number)
    ]

    plot_set_of_maps(
        cluster_centers,
        titles,
        "Empirical cluster centers",
        plot_reordering=plot_reordering,
        **kwargs,
    )


def plot_spatial_odds_ratio(
    target_binary_with_labels: xr.DataArray,
    plot_reordering: np.ndarray | None = None,
    vmax: int = 5,
    **kwargs,
):
    """
    Plot, for each cluster and at each grid point, the mean of a binary target within the 
    cluster divided by the mean of that target over all times. This corresponds to the 
    odds ratio of the target within each cluster.

    :param target_binary_with_labels: The target data (e.g., exceedance of a precipitation 
                                      threshold) for all time steps. Must have dimensions 
                                      "time", "latitude", and "longitude". The time dimension 
                                      must have a coordinate named "label" that contains the 
                                      cluster labels for each time step.
    :param plot_reordering: A list of indices to reorder the clusters for plotting.
    :param vmax: used to set the contour levels, which will be [1/vmax, 1/(vmax-1), ... . vmax-1, vmax)]
    :param kwargs:          Other arguments, passed to the contourf function for plotting.
    """
    target_mean_by_cluster = target_binary_with_labels.groupby("label").mean()
    target_mean_all = target_binary_with_labels.mean("time")
    odds_ratio = target_mean_by_cluster / target_mean_all

    levels = np.concatenate(
        [1 / np.arange(2, vmax + 1, 1)[::-1], np.arange(1, vmax + 1, 1)]
    )

    cluster_number = odds_ratio.values.shape[0]
    titles = [f"Cluster {i}" for i in range(cluster_number)]

    plot_set_of_maps(
        odds_ratio,
        titles,
        "Odds ratio of target variable within each cluster",
        plot_reordering=plot_reordering,
        levels=levels,
        **kwargs,
    )


def plot_reordered_centers_and_odds_ratio(
    vae: VAE, inputs: xr.DataArray, target_binary: xr.DataArray
):
    """
    Plot a summary of the CMM-VAE clusters characteristics: centers (both empirical and 
    decoded) and odds ratio of a binary target.
    The clusters are ordered, for plotting, by the mean of the target within each cluster.

    :param vae: The CMM-VAE used to calculate cluster centers
    :param inputs: The input data (e.g., z500) for all time steps. Must have dimensions 
                   "time", "latitude", and "longitude".
    :param target_binary: The target data (e.g., exceedance of a precipitation threshold) 
                          for all time steps. Must have the same shape and dimensions as "inputs".
    """

    labels = get_input_labels(vae, inputs.values)
    inputs_with_label = inputs.assign_coords(label=("time", labels))
    target_binary_with_label = target_binary.assign_coords(label=("time", labels))

    # Reorder clusters by mean target for plotting
    mean_target_by_label = (
        target_binary_with_label.mean(("latitude", "longitude")).groupby("label").mean()
    )
    label_reordering = mean_target_by_label.argsort().values

    plot_cluster_centers(
        vae,
        inputs[0],
        plot_reordering=label_reordering,
        levels=np.arange(-2.0, 2.1, 0.25),
    )
    plot_empirical_cluster_centers(
        inputs_with_label,
        plot_reordering=label_reordering,
        levels=np.arange(-2.0, 2.1, 0.25),
    )
    plot_spatial_odds_ratio(
        target_binary_with_label,
        plot_reordering=label_reordering,
        cmap="PuOr",
        extend="both",
    )
