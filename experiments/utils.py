import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy import ndarray
import cartopy.feature as cfeature


def filter_dataset(
    dataset: xr.Dataset,
    latitude: tuple[int, int] | None,
    longitude: tuple[int, int] | None,
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
                                    ``"north atlantic"``, ``"cca"``, ``"new atlantic"``, and ``"global"``.
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
    elif geographical_filter == "global":
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
    data = np.reshape(data, [nt, ny * nx])

    return data


def plot_losses(training_loss: np.ndarray, validation_loss: np.ndarray):
    """
    Plot the training loss and validation loss.

    :param training_loss:   The training loss.
    :param validation_loss: The validation loss.
    """
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    plt.title(label="Model Loss by Epoch", loc="center")
    ax.plot(training_loss, label="Training Data", color="gray")
    ax.plot(validation_loss, label="Test Data", color="red")
    ax.set(xlabel="Epoch", ylabel="Loss")
    plt.legend()
    plt.show()
