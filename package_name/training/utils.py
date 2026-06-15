from tensorflow.keras import backend as K
import tensorflow as tf
from dataclasses import dataclass
from typing import Callable


@dataclass
class Config:
    latent_dim: int
    dim_layer1: int
    dim_layer2: int
    dim_layer3: int
    activation: str


@dataclass
class EncoderConfig(Config):
    original_dim: int
    original_dim_target: int
    cluster_number: int
    pr_cluster_number: int
    sampling_fn: Callable


# This class is the same as EncoderConfig but was duplicated for code readability
@dataclass
class VAEConfig(Config):
    original_dim: int
    original_dim_target: int
    cluster_number: int
    pr_cluster_number: int
    sampling_fn: Callable


@dataclass
class DecoderConfig(Config):
    original_dim: int


@dataclass
class EncoderInput:
    x: tf.Tensor  # (batch, *input_shape)
    dummy: tf.Tensor  # (batch, 1)
    target: tf.Tensor  # (batch, *input_shape)


@dataclass
class LatentSpaceOutput:
    z_mean: tf.Tensor  # (batch, latent_dim)
    z_log_var: tf.Tensor  # (batch, latent_dim)
    z: tf.Tensor  # (batch, latent_dim)


@dataclass
class MixtureOutput:
    mu: tf.Tensor  # (batch, cluster_number, latent_dim)
    pi: tf.Tensor  # (batch, cluster_number)


@dataclass
class ClustersOutput:
    clusters_pred: tf.Tensor  # (batch, cluster_number)
    target_pred: tf.Tensor  # (batch, pr_cluster_number)
    target_clusters_pred: tf.Tensor  # (batch, cluster_number)


@dataclass
class EncoderOutput:
    latent: LatentSpaceOutput
    mixture: MixtureOutput
    clusters: ClustersOutput


@dataclass
class DecoderOutput:
    x_recon: tf.Tensor  # (batch, *input_shape)


@dataclass
class Loss:
    vae_reconstruction: tf.Tensor  # (1,)
    vae_regularisation: tf.Tensor  # (1,)
    target_prediction: tf.Tensor  # (1,)
    cluster_target_regularisation: tf.Tensor  # (1,)
    mixture_regularization: tf.Tensor  # (1,)

    @property
    def total(self):
        return (
            self.vae_reconstruction
            + self.vae_regularisation
            + self.target_prediction
            + self.cluster_target_regularisation
            + self.mixture_regularization
        )


@dataclass
class GaussianDistribution:
    mean: tf.Tensor
    log_var: tf.Tensor


def format_encoder_output_as_object(encoder_output: dict) -> EncoderOutput:
    """
    Format the encoder output as an object.

    :param encoder_output:  The encoder output to format.
    :return:                The formatted output.
    """
    latent_output = LatentSpaceOutput(
        z=encoder_output["latent"]["z"],
        z_mean=encoder_output["latent"]["z_mean"],
        z_log_var=encoder_output["latent"]["z_log_var"],
    )

    mixture_output = MixtureOutput(
        mu=encoder_output["mixture"]["mu"],
        pi=encoder_output["mixture"]["pi"],
    )

    clusters_output = ClustersOutput(
        clusters_pred=encoder_output["clusters"]["clusters_pred"],
        target_pred=encoder_output["clusters"]["target_pred"],
        target_clusters_pred=encoder_output["clusters"]["target_clusters_pred"],
    )

    encoder_output = EncoderOutput(
        latent=latent_output,
        mixture=mixture_output,
        clusters=clusters_output,
    )

    return encoder_output


def format_decoder_output_as_object(
    decoder_output: dict,
) -> DecoderOutput:
    """
    Format the decoder output as an object.

    :param decoder_output:  The decoder output to format.
    :return:                The formatted output.
    """
    decoder_output = DecoderOutput(x_recon=decoder_output["x_recon"])
    return decoder_output


def format_encoder_input_as_object(
    encoder_input: dict,
) -> EncoderInput:
    """
    Format the encoder input as an object.

    :param encoder_input:  The encoder input to format.
    :return:               The formatted input.
    """
    encoder_input = EncoderInput(
        x=encoder_input["x"],
        dummy=encoder_input["dummy"],
        target=encoder_input["target"],
    )
    return encoder_input


def sampling(args: tuple[tf.Tensor, float]) -> tf.Tensor:
    """
    Sample a latent vector from a Gaussian distribution using the reparameterization trick.

    :param args: Tuple ``(z_mean, z_log_sigma)`` containing the latent mean and log-variance tensors.
    :return:     A sampled latent tensor with the same shape as ``z_mean``.
    """
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(
        shape=(K.shape(z_mean)[0], K.int_shape(z_mean)[1]), mean=0.0, stddev=1.0
    )
    return z_mean + K.exp(0.5 * z_log_sigma) * epsilon
