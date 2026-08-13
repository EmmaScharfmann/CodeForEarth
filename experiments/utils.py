import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy import ndarray
import cartopy.feature as cfeature
from typing import Sequence
from package_name.constants import EPSILON
from enum import Enum

class GeographicalFilter(Enum):
    """Enum for supported geographical filters."""
    MEDITERRANEAN = "mediterranean"
    MOROCCO = "morocco"
    LARGER_MEDITERRANEAN = "larger mediterranean"
    ATLANTIC = "atlantic"
    NORTH_ATLANTIC = "north atlantic"
    CCA = "cca"
    NEW_ATLANTIC = "new atlantic"
    EXTENDED_EUROPE = "extended europe"
    GLOBAL = "global"
 
 
GEOGRAPHICAL_BOUNDS = {
    GeographicalFilter.MEDITERRANEAN: ((25, 50), (-20, 45)),
    GeographicalFilter.MOROCCO: ((30, 36), (-11, 0)),
    GeographicalFilter.LARGER_MEDITERRANEAN: ((25, 60), (-30, 45)),
    GeographicalFilter.ATLANTIC: ((25, 80), (-50, 30)),
    GeographicalFilter.NORTH_ATLANTIC: ((50, 65), (-60, 0)),
    GeographicalFilter.CCA: ((20, 50), (-30, 20)),
    GeographicalFilter.NEW_ATLANTIC: ((20, 80), (-50, 30)),
    GeographicalFilter.EXTENDED_EUROPE: ((25, 80), (-20, 40)),
    GeographicalFilter.GLOBAL: (None, None),
}

def filter_dataset(
    dataset: xr.Dataset,
    geographical_filter: GeographicalFilter,
) -> xr.Dataset:
    """
    Filter a dataset by latitude and longitude.
    
    :param dataset:              The dataset to be filtered.
    :param geographical_filter:  The geographical filter to be applied.
    :return:                     The filtered dataset.
    """
    bounds = GEOGRAPHICAL_BOUNDS[geographical_filter]
    
    dataset = dataset.sortby("latitude")
    dataset = dataset.sortby("longitude")
    
    if geographical_filter == GeographicalFilter.GLOBAL:
        return dataset

    bounds = GEOGRAPHICAL_BOUNDS[geographical_filter]
    return dataset.sel(
        latitude=slice(bounds[0][0], bounds[0][1]),
        longitude=slice(bounds[1][0], bounds[1][1]),
    )

def _generate_spatial_coordinates(
    lower_bound: float,
    upper_bound: float,
    resolution: float,
) -> np.ndarray:
    """Generate regularly spaced coordinates inside two bounds."""
    if resolution <= 0:
        raise ValueError("Spatial resolution must be greater than zero.")

    lower_bound = min(lower_bound, upper_bound)
    upper_bound = max(lower_bound, upper_bound)

    number_of_steps = int(
        np.floor(
            (upper_bound - lower_bound) / resolution + EPSILON
        )
    )

    coordinates = (
        lower_bound
        + np.arange(number_of_steps + 1) * resolution
    )
    return np.round(coordinates, decimals=10)


def _change_spatial_resolution(
    dataset: xr.Dataset | xr.DataArray,
    geographical_filter: GeographicalFilter,
    spatial_resolution: float,
    interpolation_method: str = "nearest",
) -> xr.Dataset | xr.DataArray:
    """
    Filter and interpolate data to a requested spatial resolution.
    
    dataset: dataset to be filtered and interpolated
    geographical_filter: name of the predefined geographical region to retain.
    spatial_resolution: requested spatial resolution in degrees.
    interpolation_method: method to use for interpolation. Default is 'nearest'.
    
    returns: filtered and interpolated dataset
    """

    if spatial_resolution <= 0:
        raise ValueError("Spatial resolution must be greater than zero.")

    dataset = filter_dataset(
        dataset=dataset,
        geographical_filter=geographical_filter,
    )

    if geographical_filter == GeographicalFilter.GLOBAL:
        latitude_bounds, longitude_bounds = _get_avaialble_coordinate_bounds(dataset)
    else:
        latitude_bounds, longitude_bounds = GEOGRAPHICAL_BOUNDS[
            geographical_filter
        ]

    target_lats = _generate_spatial_coordinates(
        lower_bound=min(latitude_bounds),
        upper_bound=max(latitude_bounds),
        resolution=spatial_resolution,
    )

    target_lons = _generate_spatial_coordinates(
        lower_bound=min(longitude_bounds),
        upper_bound=max(longitude_bounds),
        resolution=spatial_resolution,
    )

    current_lats = np.asarray(dataset.latitude.values)
    current_lons = np.asarray(dataset.longitude.values)

    latitude_matches = (
        len(current_lats) == len(target_lats)
        and np.allclose(current_lats, target_lats)
    )

    longitude_matches = (
        len(current_lons) == len(target_lons)
        and np.allclose(current_lons, target_lons)
    )

    if latitude_matches is True and longitude_matches is True:
        return dataset

    return dataset.interp(
        latitude=target_lats,
        longitude=target_lons,
        method=interpolation_method,
        kwargs={"fill_value": "extrapolate"},
    )
    
def _get_avaialble_coordinate_bounds(
    dataset: xr.Dataset | xr.DataArray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """For the global region, use the available coordinate bounds.
    """
    latitude_bounds = (
        float(dataset.latitude.min()),
        float(dataset.latitude.max()),
    )
    longitude_bounds = (
        float(dataset.longitude.min()),
        float(dataset.longitude.max()),
    )
    return latitude_bounds, longitude_bounds

def _calculate_anomalies(
    x: xr.Dataset | xr.DataArray,
    dim: str | Sequence[str] = "time",
) -> xr.Dataset | xr.DataArray:
    """
    Calculate anomalies by removing the mean over one or more dimensions.
    """
    return x - x.mean(dim=dim)


def preprocess_dataset(
    filename: str,
    variable_name: str,
    multiplication_factor: float,
    geographical_filter: GeographicalFilter,
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
    :param months_filter:           List of months to retain, expressed as integers between 1 and 12.
    :param anomalies:               If True, remove the mean for each day of the year, producing daily anomalies.
    :param normalization:           If True, divide the data by its standard deviation over the time dimension.
    :param rolling_window:          Size of the centered rolling mean window along the time dimension. If 0, no smoothing is applied.
    :return:                        Preprocessed data array containing the selected variable.
    """

    dataset = xr.open_dataset(filename)[variable_name] * multiplication_factor
    dataset = filter_dataset(dataset=dataset, geographical_filter=geographical_filter)
    dataset = dataset.sel(time=np.isin(dataset.time.dt.month, months_filter))

    if anomalies:
        dataset = dataset.groupby("time.dayofyear").map(_calculate_anomalies)
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

    
def preprocess_forecast_data(
    dataset: xr.Dataset,
    variable_name: str,
    multiplication_factor: float,
    geographical_filter: GeographicalFilter,
    anomalies: bool,
    normalization: bool,
    rolling_window: int,
    spatial_resolution: float,
    weights: xr.DataArray | None = None,
) -> xr.Dataset:
    """Load and preprocess forecast data for one reference date.
    
    :param dataset:               Dataset containing the forecast data (members and control merged).
    :param variable_name:         Name of the variable to extract from the dataset.
    :param multiplication_factor: Factor by which to multiply the variable values after loading.
    :param geographical_filter:   Name of the predefined geographical region to retain. 
    :param anomalies:             If True, remove the mean for each day of the year, producing daily anomalies.
    :param normalization:         If True, divide the data by its standard deviation over the time dimension.
    :param rolling_window:        Size of the centered rolling mean window along the time dimension. If 0, no smoothing is applied.
    :param spatial_resolution:    Spatial resolution in degrees to which the data should be interpolated after preprocessing.
    :param weights:               Optional DataArray of weights to apply to the dataset after preprocessing.
    :return:                      Preprocessed dataset containing the selected variable.
    """

    dataset[variable_name] *= multiplication_factor

    # Convert forecast steps to daily lead times.
    dataset = dataset.groupby(
        dataset["step"].dt.days
    ).mean()

    if anomalies:
        dataset = _calculate_anomalies(
            dataset,
            dim=("time", "number"),
        )

    if normalization:
        standard_deviation = dataset.std()
        dataset = dataset / standard_deviation.where(
            standard_deviation != 0
        )

    dataset = _change_spatial_resolution(
        dataset=dataset,
        geographical_filter=geographical_filter,
        spatial_resolution=spatial_resolution,
        interpolation_method="nearest",
    )
    
    if rolling_window < 0:
        raise ValueError(
            "rolling_window cannot be negative."
        )
        
    dataset = dataset.rolling(days=rolling_window,
                                min_periods=1,
                                center=True,).mean()

    if weights is not None:
        dataset = dataset * weights

    dataset.load()

    return dataset


def visualise_s2s_cluster_contours(
    dataset_xarray: xr.DataArray,
    regime_names: list[str],
    regime_order: list[int],
    vmin: float,
    vmax: float,
    step: float,
    color_scheme: str,
    unit: str | None = None,
    col_number: int = 2,
    borders: bool = True,
    projection=ccrs.Orthographic(0, 45),
):
    """
    Visualise S2S cluster-center fields as contour maps.

    :param dataset_xarray: DataArray with dimensions ``label``, ``latitude``, and ``longitude``.
    :param regime_names: Names to use as subplot titles.
    :param regime_order: Order in which the regimes should be displayed.
    :param vmin: Minimum contour value.
    :param vmax: Maximum contour value.
    :param step: Contour interval.
    :param color_scheme: Matplotlib colormap name.
    :param unit: Unit label for the colorbar.
    :param col_number: Number of subplot columns.
    :param borders: Whether to draw country borders.
    :param projection: Cartopy projection used for the subplot axes.
    :return: Matplotlib figure.
    """
    x, y = np.meshgrid(
        dataset_xarray.longitude,
        dataset_xarray.latitude,
    )

    fig, axes = plt.subplots(
        1,
        col_number,
        figsize=(14, 5),
        subplot_kw={"projection": projection},
    )

    levels = np.arange(vmin, vmax, step)

    for plot_index, regime_index in enumerate(regime_order):

        cs = axes.flat[plot_index].contourf(
            x,
            y,
            dataset_xarray[regime_index, :, :],
            levels=levels,
            transform=ccrs.PlateCarree(),
            cmap=color_scheme,
            extend="both",
        )

        axes.flat[plot_index].coastlines()

        if borders:
            axes.flat[plot_index].add_feature(cartopy.feature.BORDERS)

        axes.flat[plot_index].set_title(regime_names[plot_index])

    cbar = fig.colorbar(
        cs,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.05,
        pad=0.08,
    )

    if unit is not None:
        cbar.set_label(unit)

    plt.tight_layout()

    return fig

