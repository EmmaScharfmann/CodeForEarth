from tensorflow import Tensor
from tensorflow.keras.losses import mse
import tensorflow as tf
from package_name.model.utils import LossFactorsConfig

from package_name.model.utils import (
    Loss,
    EncoderInput,
    DecoderOutput,
    ClustersOutput,
    LatentSpaceOutput,
    MixtureOutput,
    EncoderOutput,
    GaussianDistribution,
)

_EPSILON = tf.keras.backend.epsilon()


class VAELoss:

    def __init__(
        self,
        original_dim: int,
        pr_cluster_number: int,
        loss_factors: LossFactorsConfig,
    ):
        self.original_dim = original_dim
        self.pr_cluster_number = pr_cluster_number
        self.loss_factors = loss_factors

    def compute(
        self,
        encoder_input: EncoderInput,
        encoder_output: EncoderOutput,
        decoder_output: DecoderOutput,
    ) -> Loss:
        """
        Compute the loss function of the given inputs and outputs.

        :param encoder_input:    The model output data.
        :param encoder_output:    The model input data.
        :param decoder_output:    The model output data.
        :return:                  The loss function, separated into different logical parts.
        """
        return Loss(
            vae_reconstruction=self._calculate_vae_reconstruction_loss(
                encoder_input=encoder_input, decoder_output=decoder_output
            ),
            vae_regularisation=_calculate_vae_regularisation_loss(
                mixture_output=encoder_output.mixture,
                clusters_output=encoder_output.clusters,
                latent_space_output=encoder_output.latent,
            ),
            target_prediction=self._calculate_target_prediction_loss(
                encoder_input=encoder_input, clusters_output=encoder_output.clusters
            ),
            cluster_target_regularisation=_calculate_cluster_target_regularisation_loss(
                clusters_output=encoder_output.clusters
            ),
            mixture_regularization=self._calculate_mixture_regularisation(
                clusters_output=encoder_output.clusters,
                mixture_output=encoder_output.mixture,
            ),
            vae_reconstruction_loss_factor=self.loss_factors.vae_reconstruction_loss_factor,
            vae_regularisation_loss_factor=self.loss_factors.vae_regularisation_loss_factor,
            target_prediction_loss_factor=self.loss_factors.target_prediction_loss_factor,
            cluster_target_regularisation_loss_factor=self.loss_factors.cluster_target_regularisation_loss_factor,
            mixture_regularization_loss_factor=self.loss_factors.mixture_regularization_loss_factor,
            dirichlet_loss_factor=self.loss_factors.dirichlet_loss_factor,
        )

    def _calculate_vae_reconstruction_loss(
        self, encoder_input: EncoderInput, decoder_output: DecoderOutput
    ) -> Tensor:
        """Calculate the reconstruction loss of the given inputs and outputs."""
        return (
            mse(encoder_input.x, decoder_output.x_recon)
            * self.original_dim  # TODO: why * self.original_dim?
        )

    def _calculate_target_prediction_loss(
        self, encoder_input: EncoderInput, clusters_output: ClustersOutput
    ) -> Tensor:
        """Calculate the component of the loss corresponding to the target prediction (as a KL divergence)"""
        target_true = encoder_input.target
        target_pred = clusters_output.target_pred

        return _calculate_categorical_kl_divergence(
            p=target_true, q=target_pred
        ) * tf.constant(  # TODO: why * self.pr_cluster_number?
            self.pr_cluster_number,
            dtype=target_true.dtype,
        ) 

    def _calculate_mixture_regularisation(
        self, clusters_output: ClustersOutput, mixture_output: MixtureOutput
    ) -> Tensor:
        """Calculate the component of the loss which regularized the mixture distribution."""
        # Categorical KL: match cluster assignment by the encoder with the cluster probability of the mixture model
        mc = tf.reduce_mean(clusters_output.clusters_pred, axis=0)
        mpi = tf.reduce_mean(mixture_output.pi, axis=0)
        categorical_KL_divergence = _calculate_categorical_kl_divergence(p=mc, q=mpi)

        # Dirichlet prior term: regularized mixture weights to avoid vanishing clusters
        dirichlet_loss = tf.reduce_sum(
            -self.loss_factors.dirichlet_loss_factor
            * tf.math.log(tf.maximum(mixture_output.pi, _EPSILON)),
            axis=-1,
        )

        return categorical_KL_divergence + dirichlet_loss


def _calculate_vae_regularisation_loss(
    latent_space_output: LatentSpaceOutput,
    mixture_output: MixtureOutput,
    clusters_output: ClustersOutput,
) -> Tensor:
    """Calculate VAE regularisation loss in order to align latent space with cluster's center"""
    z_mean = tf.expand_dims(latent_space_output.z_mean, axis=1)
    z_log_var = tf.expand_dims(latent_space_output.z_log_var, axis=1)

    mu = mixture_output.mu
    log_var_prior = tf.zeros_like(mu)

    component_kl = _calculate_gaussian_kl_divergence(
        p=GaussianDistribution(mean=z_mean, log_var=z_log_var),
        q=GaussianDistribution(mean=mu, log_var=log_var_prior),
    )
    gaussian_kl = tf.reduce_sum(
        component_kl * clusters_output.clusters_pred,
        axis=-1,
    ) 

    return gaussian_kl


def _calculate_cluster_target_regularisation_loss(
    clusters_output: ClustersOutput,
) -> Tensor:
    """Calculate the component of the loss ensuring that the cluster prediction made directly from the input variable (c) and that made from the target variable (cr) are close to each other"""
    return _calculate_categorical_kl_divergence(
        clusters_output.clusters_pred, clusters_output.target_clusters_pred
    ) 


def _calculate_gaussian_kl_divergence(
    p: GaussianDistribution, q: GaussianDistribution
) -> Tensor:
    """Calculate the KL divergence between the two given gaussian distributions."""
    var_p = tf.exp(p.log_var)
    var_q = tf.exp(q.log_var)
    return 0.5 * tf.reduce_sum(
        q.log_var - p.log_var + (var_p + tf.square(p.mean - q.mean)) / var_q - 1,
        axis=-1,
    )


def _calculate_categorical_kl_divergence(p: Tensor, q: Tensor) -> Tensor:
    """Calculate the KL divergence between the two given categorical distributions."""
    log_p = tf.math.log(tf.maximum(p, _EPSILON))
    log_q = tf.math.log(tf.maximum(q, _EPSILON))

    return tf.reduce_sum(p * (log_p - log_q), axis=-1)
