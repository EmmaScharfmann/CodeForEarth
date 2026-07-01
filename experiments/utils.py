import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy import ndarray
import cartopy.feature as cfeature
import tensorflow as tf



def filter_dataset(
    dataset: xr.Dataset, latitude: tuple[int, int] | None  , longitude: tuple[int, int] | None
) -> xr.Dataset:
    """
    Filter a dataset by latitude and longitude.

    :param dataset:    The dataset to be filtered.
    :param latitude:   The latitude of the dataset to be filtered. None if the dataset is not filtered on the latitude.
    :param longitude:  The longitude of the dataset to be filtered. None if the dataset is not filtered on the longitude.
    :return:           The filtered dataset.
    """
    if latitude is None and longitude is None:
        return dataset
    elif latitude is None:
        return dataset.sel(
            longitude=slice(longitude[0], longitude[1]),
        )
    elif longitude is None:
        return dataset.sel(
            latitude=slice(latitude[0], latitude[1]),
        )
    return dataset.sel(
        latitude=slice(latitude[0], latitude[1]),
        longitude=slice(longitude[0], longitude[1]),
    )


def calculate_anomalies(x: xr.Dataset) -> xr.Dataset:
    """
    Compute temporal anomalies by removing the time mean.

    :param x:   The temporal anomalies.
    :return:    The anomalies.
    """
    return x - x.mean(dim="time")


def preprocess_dataset(
    filename: str,
    variable_name: str,
    multiplication_factor: float,
    geographical_filter: str,
    months_filter: list[int],
    anomalies: bool,
    normalization: bool,
    rolling_window: int,
) -> xr.Dataset:
    """
    Load and preprocess a geospatial time series dataset.

    :param filename:                Path to the input NetCDF file.
    :param variable_name:           Name of the variable to extract from the dataset.
    :param multiplication_factor:   Factor by which to multiply the variable values after loading.
    :param geographical_filter:     Name of the predefined geographical region to retain.
                                    Supported values are:
                                    ``"mediterranean"``, ``"morocco"``,
                                    ``"larger mediterranean"``, ``"atlantic"``,
                                    ``"north atlantic"``, ``"cca"``, and ``"new atlantic"``.
    :param months_filter:           List of months to retain, expressed as integers between 1 and 12.
    :param anomalies:               If True, remove the mean for each day of the year, producing daily anomalies.
    :param normalization:           If True, divide the data by its standard deviation over the time dimension.
    :param rolling_window:          Size of the centered rolling mean window along the time dimension. If 0, no smoothing is applied.
    :return:                        Preprocessed data array containing the selected variable.
    """

    dataset = xr.open_dataset(filename)[variable_name] * multiplication_factor

    if geographical_filter == "mediterranean":
        latitude = (25, 50)
        longitude = (-20, 45)
    elif geographical_filter == "morocco":
        latitude = (36, 30)
        longitude = (-11, 0)
    elif geographical_filter == "larger mediterranean":
        latitude = (25, 60)
        longitude = (-30, 45)
    elif geographical_filter == "atlantic":
        latitude = (25, 80)
        longitude = (-50, 30)
    elif geographical_filter == "north atlantic":
        latitude = (50, 65)
        longitude = (-60, 0)
    elif geographical_filter == "cca":
        latitude = (20, 50)
        longitude = (-30, 20)
    elif geographical_filter == "new atlantic":
        latitude = (20, 80)
        longitude = (-50, 30)
    elif geographical_filter == "extended europe":
        latitude = (25, 80)
        longitude = (-20, 40)
    elif geographical_filter == 'global':
        latitude = None
        longitude = None
    else:
        raise ValueError(f"Unknown geographical filter: {geographical_filter}")

    dataset = filter_dataset(dataset=dataset, latitude=latitude, longitude=longitude)
    dataset = dataset.sel(time=np.isin(dataset.time.dt.month, months_filter))

    if anomalies:
        dataset = dataset.groupby("time.dayofyear").map(calculate_anomalies)
    if normalization:
        dataset = dataset / dataset.std(dim="time")
    if rolling_window != 0:
        dataset = dataset.rolling(
            time=rolling_window, min_periods=1, center=True
        ).mean()

    return dataset


def reshape_data_for_clustering(
    xarray_data: xr.Dataset,
) -> ndarray:
    """
    Reshape a 3D spatiotemporal data array into a 2D array suitable
    for clustering algorithms.

    :param xarray_data: The input data is assumed to have dimensions ``(time, latitude, longitude)`` (or equivalent).
    :return:            Two-dimensional array of shape ``(n_time, n_grid_points)``, where each row corresponds to a time step and each column corresponds to a spatial grid point.
    """
    data = xarray_data.values

    nt, ny, nx = data.shape
    data = np.reshape(data, [nt, ny * nx], order="F")

    return data


def plot_losses(training_loss: np.ndarray, validation_loss: np.ndarray):
    """
    Plot the training loss and validation loss.

    :param training_loss:   The training loss.
    :param validation_loss: The validation loss.
    """
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    plt.title(label="Model Loss by Epoch", loc="center")
    ax.plot(training_loss, label="Training Data", color="white")
    ax.plot(validation_loss, label="Test Data", color="red")
    ax.set(xlabel="Epoch", ylabel="Loss")
    plt.legend()
    plt.show()

def plot_all_losses(history: tf.keras.callbacks.History):
    """
    Plot the training loss and validation loss.

    :param history:   The history of the model.
    """
    #normalize the losses by the maximum value of that loss
    for key in history.history.keys():
        #cut out first 2 epochs to avoid the initial spike in loss
        history.history[key] = history.history[key][2:]
        max_value = max(history.history[key])
        if max_value != 0:
            history.history[key] = [x / max_value for x in history.history[key] ]

    x = np.arange(2, len(history.history["total_loss"])+2)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    plt.title(label="Model Loss by Epoch", loc="center")
    ax.plot(x, history.history["total_loss"], label="Total Loss", color="red", linewidth=2)
    ax.plot(x, history.history["val_total_loss"], color="red", linewidth=1)

    ax.plot(x, history.history["reconstruction_loss"], label="Reconstruction Loss", color="blue", linewidth=2)
    ax.plot(x, history.history["val_reconstruction_loss"], color="blue", linewidth=1)

    ax.plot(x, history.history["vae_regularisation_loss"], label="VAE Regularisation Loss", color="green", linewidth=2)
    ax.plot(x, history.history["val_vae_regularisation_loss"], color="green", linewidth=1)

    ax.plot(x, history.history["target_prediction_loss"], label="Target Prediction Loss", color="orange", linewidth=2)
    ax.plot(x, history.history["val_target_prediction_loss"], color="orange", linewidth=1)

    ax.plot(x, history.history["cluster_target_regularisation_loss"], label="Cluster Target Regularisation Loss", color="purple", linewidth=2)
    ax.plot(x, history.history["val_cluster_target_regularisation_loss"], color="purple", linewidth=1)

    ax.plot(x, history.history["mixture_regularization_loss"], label="Mixture Regularization Loss", color="brown", linewidth=2)
    ax.plot(x, history.history["val_mixture_regularization_loss"], color="brown", linewidth=1)
    
    ax.set(xlabel="Epoch", ylabel="Loss")
    plt.legend()
    plt.show()


def plot_cluster_centers(cluster_centers: xr.DataArray,
    labels_data: np.ndarray,
    label_reordering: np.ndarray | None = None,
    borders: bool = True,
    projection: ccrs.Projection = ccrs.Orthographic(0, 45),
    **kwargs):
    """
    Plot cluster centers on a map.

    :param cluster_centers:     The cluster centers to plot. Must have dimensions 'label', 'latitude', and 'longitude'.
    :param labels_data:         Cluster labels for each time step, used to calculate the frequency of each cluster.
    :param label_reordering:    A list of indices to reorder the clusters. the first cluster to be plotted will be label_reordering[0], the second cluster will be label_reordering[1], and so on. If None, clusters will be plotted in their original order.
    :param borders:             If True, add country borders to the map.
    :param projection:          The cartopy projection to use for the map.
    :param kwargs:              Other arguments, passed to the contourf function for plotting.
    """
    cluster_number = cluster_centers.values.shape[0]

    fig, axs = plt.subplots(1, cluster_number, figsize=(4 * cluster_number, 4), subplot_kw=dict(projection=projection))

    cluster_counts = [labels_data[labels_data == i].shape[0] for i in range(cluster_number)]
    cluster_frequencies = np.array(cluster_counts) / len(labels_data)

    if label_reordering is None:
        label_reordering = np.arange(cluster_number)

    for i, ax in enumerate(axs):
        cluster_centers[label_reordering[i]].plot.contourf(ax=ax, transform=ccrs.PlateCarree(), **kwargs)
        ax.coastlines()

        if borders:
            ax.add_feature(cfeature.BORDERS)

        title = f"Cluster {label_reordering[i]}, {cluster_frequencies[i] * 100:.1f}%"
        ax.set_title(title)
    fig.tight_layout()