import numpy as np
import xarray as xr
from package_name.inference.predictor import VAEPredictor
from package_name.data_processing.data_processor import flatten_input


def predict_clusters(
    vae: VAEPredictor,
    inputs: np.ndarray,
) -> np.ndarray:
    """
    From a set of input data (e.g., Z500), calculate cluster assignment probabilities
    for each time step using a CMM-VAE.

    :param inputs: Input data to be labeled, with shape (# times, # latitudes, # longitudes)
    :param vae: The CMM-VAE used to predict cluster assigments
    :return: An array of cluster assignment probabilities for each time step, with shape
    (# times, vae.cfg.cluster_number)
    """
    encoder_input = flatten_input(inputs)
    encoder_output = vae.encode(encoder_input)
    cluster_probabilities = encoder_output["clusters_pred"]

    return cluster_probabilities


def calculate_cluster_centers(vae: VAEPredictor) -> np.ndarray:
    """
    Get cluster centers in input space by decoding the means of the mixture components
    in latent space.

    :param vae: The CMM-VAE used to calculate cluster centers
    :return: The cluster centers in input space (array of shape
        (vae.cfg.cluster_number, vae.cfg.original_dim))
    """
    mu = vae.get_mixture_components()["mu"][0]
    return vae.decode(mu)["x_recon"]
