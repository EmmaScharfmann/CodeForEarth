from tensorflow import Tensor
from tensorflow.keras.losses import mse, categorical_crossentropy
import tensorflow as tf

from package_name.training.utils import (
    Loss,
    EncoderInput,
    DecoderOutput,
    AuxiliaryOutput,
    LatentSpace,
    MixtureComponents,
    EncoderOutput,
)


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
            reconstruction=self._calculate_reconstruction_loss(
                encoder_input=encoder_input, decoder_output=decoder_output
            ),
            prediction=_cluster_prediction(auxiliary_output=encoder_output.aux),
            target=self._target(
                encoder_input=encoder_input, auxiliary_output=encoder_output.aux
            ),
            kl_gaussian=_kl_gaussian(
                latent_space_output=encoder_output.latent,
                mixture_output=encoder_output.mixture,
            ),
            kl_categorical=_kl_categorical(
                mixture_output=encoder_output.mixture,
                auxiliary_output=encoder_output.aux,
            ),
            dirichlet=_dirichlet(mixture_output=encoder_output.mixture),
        )

    def _calculate_reconstruction_loss(
        self, encoder_input: EncoderInput, decoder_output: DecoderOutput
    ) -> Tensor:
        """Calculate the reconstruction loss of the given inputs and outputs."""
        return (
            mse(encoder_input.x, decoder_output.x_recon)
            * self.original_dim
            * self.reconstruction_loss_factor
        )

    def _target(
        self, encoder_input: EncoderInput, auxiliary_output: AuxiliaryOutput
    ) -> Tensor:
        """Calculate the target of the given inputs and outputs."""
        r_true = tf.cast(encoder_input.r, tf.float32)
        r_pred = auxiliary_output.r

        return (
            categorical_crossentropy(r_true, r_pred)
            + categorical_crossentropy(r_true, r_true)
        ) * self.pr_cluster_number


def _cluster_prediction(auxiliary_output: AuxiliaryOutput) -> Tensor:
    """Calculate the categorical cross-entropy for the given auxiliary output."""
    return categorical_crossentropy(auxiliary_output.c, auxiliary_output.cr)


def _kl_categorical(
    mixture_output: MixtureComponents, auxiliary_output: AuxiliaryOutput
) -> Tensor:
    """Calculate the KL loss for the categorical components."""
    c = auxiliary_output.c
    pi = mixture_output.pi

    mc = tf.reduce_mean(c, axis=0)
    mpi = tf.reduce_mean(pi, axis=0)

    return tf.reduce_sum(mc * tf.math.log(mc) - mc * tf.math.log(mpi))


def _dirichlet(mixture_output: MixtureComponents) -> Tensor:
    """Calculate the dirichlet loss for the given mixture components and auxiliary output."""
    pi = mixture_output.pi
    dir_prior = -0.5 * tf.math.log(pi)
    return tf.reduce_sum(dir_prior, axis=-1)


def _kl_gaussian(
    latent_space_output: LatentSpace, mixture_output: MixtureComponents
) -> Tensor:
    """Calculate the KL loss for the gaussian components."""
    z_mean = latent_space_output.z_mean
    z_log_var = latent_space_output.z_log_var
    mu = mixture_output.mu
    c = mixture_output.pi

    z_mean = tf.expand_dims(z_mean, 1)
    z_log_var = tf.expand_dims(z_log_var, 1)

    diff = z_mean - mu

    kl = 1 + z_log_var - tf.square(diff) - tf.exp(z_log_var)
    kl = tf.reduce_sum(kl, axis=-1)

    return -0.5 * tf.reduce_sum(kl * c, axis=-1)
