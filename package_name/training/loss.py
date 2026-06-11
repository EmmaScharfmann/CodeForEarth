from tensorflow import Tensor
from tensorflow.keras.losses import mse
import tensorflow as tf

from package_name.training.utils import (
    Loss,
    EncoderInput,
    DecoderOutput,
    AuxiliaryOutput,
    LatentSpace,
    MixtureComponents,
    EncoderOutput,
    GaussianDistribution,
)

_EPSILON = tf.keras.backend.epsilon()


class VAELoss:
    def __init__(
        self,
        original_dim: int,
        pr_cluster_number: int,
        reconstruction_loss_factor: float,
    ):
        self.original_dim = original_dim
        self.pr_cluster_number = pr_cluster_number
        self.reconstruction_loss_factor = reconstruction_loss_factor

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
                auxiliary_output=encoder_output.aux,
                latent_space_output=encoder_output.latent,
            ),
            target_prediction=self._calculate_target_prediction_loss(
                encoder_input=encoder_input, auxiliary_output=encoder_output.aux
            ),
            cluster_target_regularisation=_calculate_cluster_target_regularisation_loss(
                auxiliary_output=encoder_output.aux
            ),
            mixture_regularization=_calculate_mixture_regularisation(
                auxiliary_output=encoder_output.aux,
                mixture_output=encoder_output.mixture,
            ),
        )

    def _calculate_vae_reconstruction_loss(
        self, encoder_input: EncoderInput, decoder_output: DecoderOutput
    ) -> Tensor:
        """Calculate the reconstruction loss of the given inputs and outputs."""
        return (
            mse(encoder_input.x, decoder_output.x_recon)
            * self.original_dim  # TODO: why * self.original_dim?
            * self.reconstruction_loss_factor
        )

    def _calculate_target_prediction_loss(
        self, encoder_input: EncoderInput, auxiliary_output: AuxiliaryOutput
    ) -> Tensor:
        """Calculate the target of the given inputs and outputs."""
        r_true = tf.cast(encoder_input.r, tf.float32)
        r_pred = auxiliary_output.r

        return _calculate_categorical_kl_divergence(
            p=r_true, q=r_pred
        ) * tf.constant(  # TODO: why * self.pr_cluster_number?
            self.pr_cluster_number,
            dtype=r_true.dtype,
        )


def _calculate_vae_regularisation_loss(
    latent_space_output: LatentSpace,
    mixture_output: MixtureComponents,
    auxiliary_output: AuxiliaryOutput,
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
        component_kl * auxiliary_output.c,
        axis=-1,
    )

    return gaussian_kl


def _calculate_mixture_regularisation(
    auxiliary_output: AuxiliaryOutput, mixture_output: MixtureComponents
) -> Tensor:
    """"""
    # Categorical KL: match cluster assignment by the encoder with the cluster probability of the mixture model
    mc = tf.reduce_mean(auxiliary_output.c, axis=0)
    mpi = tf.reduce_mean(mixture_output.pi, axis=0)
    categorical_KL_divergence = _calculate_categorical_kl_divergence(p=mc, q=mpi)

    # Dirichlet prior term: regularized mixture weights to avoid vanishing clusters
    dirichlet_loss = tf.reduce_sum(
        -0.5 * tf.math.log(tf.maximum(mixture_output.pi, _EPSILON)),
        axis=-1,
    )

    return categorical_KL_divergence + dirichlet_loss


def _calculate_cluster_target_regularisation_loss(
    auxiliary_output: AuxiliaryOutput,
) -> Tensor:
    """Calculate the categorical cross-entropy for the given auxiliary output."""
    return _calculate_categorical_kl_divergence(auxiliary_output.c, auxiliary_output.cr)


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
