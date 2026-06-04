from tensorflow.keras import backend as K
from dataclasses import dataclass
from typing import Callable

import keras
import tensorflow as tf


@dataclass
class VAEConfig:
    original_dim: int
    original_dim_r: int
    dim_layer1: int
    dim_layer2: int
    dim_layer3: int
    activation: str
    cluster_number: int
    latent_dim: int
    pr_cluster_number: int
    sampling_fn: Callable


@dataclass
class EncoderConfig:
    input_shape: tuple
    input_shape_r: tuple
    dim_layer1: int
    dim_layer2: int
    dim_layer3: int
    activation: str
    cluster_number: int
    latent_dim: int
    pr_cluster_number: int
    sampling_fn: Callable


@dataclass
class DecoderConfig:
    latent_dim: int
    dim_layer1: int
    dim_layer2: int
    dim_layer3: int
    activation: str
    output_dim: int


@dataclass
class EncoderInput:
    x: keras.KerasTensor
    dummy: keras.KerasTensor
    r: keras.KerasTensor


@dataclass
class LatentSpace:
    z_mean: keras.KerasTensor
    z_log_var: keras.KerasTensor
    z: keras.KerasTensor


@dataclass
class MixtureComponents:
    mu: tf.Tensor
    pi: tf.Tensor


@dataclass
class AuxiliaryOutput:
    c: tf.Tensor
    r: tf.Tensor
    cr: tf.Tensor


@dataclass
class EncoderOutput:
    latent: LatentSpace
    mixture: MixtureComponents
    aux: AuxiliaryOutput


@dataclass
class DecoderOutput:
    x_recon: keras.KerasTensor


@dataclass
class Loss:
    reconstruction: tf.Tensor
    prediction: tf.Tensor
    target: tf.Tensor
    kl_gaussian: tf.Tensor
    kl_categorical: tf.Tensor
    dirichlet: tf.Tensor

    @property
    def total(self):
        return (
            self.reconstruction
            + self.prediction
            + self.target
            + self.kl_gaussian
            + self.kl_categorical
            + self.dirichlet
        )



def format_encoder_output_as_object(encoder_output: dict) -> EncoderOutput:
    latent_input = LatentSpace(
        z=encoder_output["latent"]["z"],
        z_mean=encoder_output["latent"]["z_mean"],
        z_log_var=encoder_output["latent"]["z_log_var"],
    )

    mixture_input = MixtureComponents(
        mu=encoder_output["mixture"]["mu"],
        pi=encoder_output["mixture"]["pi"],
    )

    aux_input = AuxiliaryOutput(
        c=encoder_output["aux"]["c"],
        r=encoder_output["aux"]["r"],
        cr=encoder_output["aux"]["cr"],
    )

    encoder_output = EncoderOutput(
        latent=latent_input,
        mixture=mixture_input,
        aux=aux_input,
    )

    return encoder_output


def format_decoder_output_as_object(
    decoder_output: dict,
) -> DecoderOutput:
    decoder_output = DecoderOutput(x_recon=decoder_output["x_recon"])
    return decoder_output


def format_encoder_input_as_object(
    encoder_input: dict,
) -> EncoderInput:
    decoder_output = EncoderInput(
        x=encoder_input["x"], dummy=encoder_input["dummy"], r=encoder_input["r"]
    )
    return decoder_output


def sampling(args):
    """TODO"""
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(
        shape=(K.shape(z_mean)[0], K.int_shape(z_mean)[1]), mean=0.0, stddev=1.0
    )
    return z_mean + K.exp(0.5 * z_log_sigma) * epsilon
