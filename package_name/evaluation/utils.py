import numpy as np
import xarray as xr
from package_name.inference.predictor import VAEPredictor
from package_name.data_processing.data_processor import flatten_input


def _predict_clusters_from_encoder_output(encoder_output: dict):
    """
    Gets the label of the most likely cluster from the output of the CMM-VAE encoder.
    """
    cluster_probs = encoder_output["clusters"]["clusters_pred"]
    return np.argmax(cluster_probs, axis=1)


def predict_clusters(
    vae: VAEPredictor,
    inputs: np.ndarray,
) -> np.ndarray:
    """Label a set of input data (e.g., Z500) using the most probable cluster predicted by a CMM-VAE.

    :param inputs: Input data to be labeled, with shape (# times, # latitudes, # longitudes)
    :param vae: The CMM-VAE used to predict cluster assigments
    :return: An array of cluster labels for each time step, with shape (# times) and
             values in {0, 1, ..., vae.cfg.cluster_number-1}
    """
    inputs_flattened = flatten_input(inputs)
    encoder_input = {
        "x": inputs_flattened,
        "dummy": np.ones((inputs_flattened.shape[0], 1)),
        "target": np.zeros((inputs_flattened.shape[0], vae.cfg.pr_cluster_number)),
    }
    encoder_output = vae.encode(encoder_input)
    cluster_labels = _predict_clusters_from_encoder_output(encoder_output)

    return cluster_labels


def calculate_cluster_centers(vae: VAEPredictor) -> np.ndarray:
    """
    Get cluster centers in input space by decoding the means of the mixture components in latent space.

    :param vae: The CMM-VAE used to calculate cluster centers
    :return: The cluster centers in input space (array of shape (vae.cfg.cluster_number, vae.cfg.original_dim))
    """
    dummy_input = {
        "x": np.zeros((1, vae.cfg.original_dim)),
        "dummy": np.ones((1, 1)),
        "target": np.zeros((1, vae.cfg.pr_cluster_number)),
    }
    encoded = vae.encode(dummy_input)
    mu = encoded["mixture"]["mu"][0]
    return vae.decode(mu)["x_recon"]
