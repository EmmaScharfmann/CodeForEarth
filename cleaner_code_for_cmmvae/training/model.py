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
