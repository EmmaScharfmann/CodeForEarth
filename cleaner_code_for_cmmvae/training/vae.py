import numpy as np

from cleaner_code_for_cmmvae.training.decoder import DecoderBuilder
from cleaner_code_for_cmmvae.training.encoder import EncoderBuilder
from cleaner_code_for_cmmvae.training.loss import VAELoss
from cleaner_code_for_cmmvae.training.model import (
    VAEConfig,
    EncoderConfig,
    DecoderConfig,
    EncoderInput,
    DecoderOutput,
    EncoderOutput,
    Loss,
)

from tensorflow.keras.models import Model
import tensorflow as tf

from cleaner_code_for_cmmvae.training import utils


class VAEModel(Model):
    def __init__(self, encoder: Model, decoder: Model, custom_loss: VAELoss, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.custom_loss = custom_loss

    def call(self, encoder_input: dict, training: bool = False, **kwargs):
        """
        Get the encoder output and decoder output from the encoder input.

        :param encoder_input:   The encoder input.
        :param training:        Whether the model is in training mode.
        :param kwargs:          Other arguments, to match the parent function `call`.
        :return:                A dictionary with the encoder output and decoder output.
        """
        encoder_outputs = self.encoder(encoder_input)
        z = encoder_outputs["latent"]["z"]
        decoded = self.decoder(z)
        return {
            "encoder_output": {
                "latent": {
                    "z": z,
                    "z_mean": encoder_outputs["latent"]["z_mean"],
                    "z_log_var": encoder_outputs["latent"]["z_log_var"],
                },
                "mixture": encoder_outputs["mixture"],
                "aux": encoder_outputs["aux"],
            },
            "decoder_output": {"x_recon": decoded["x_recon"]},
        }

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            losses = self._calculate_loss(inputs=inputs, training=True)
            total_loss = tf.reduce_mean(losses.total)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return _format_losses(total_loss=total_loss, losses=losses)

    def test_step(self, inputs):
        losses = self._calculate_loss(inputs=inputs, training=False)
        total_loss = tf.reduce_mean(losses.total)
        return _format_losses(total_loss=total_loss, losses=losses)

    def _calculate_loss(self, inputs: dict, training: bool = False) -> Loss:
        """Calculate the loss for the model."""
        outputs = self(inputs, training=training)
        encoder_input, encoder_output, decoder_output = _format_input_and_output(
            inputs=inputs, outputs=outputs
        )
        losses = self.custom_loss.compute(
            encoder_input=encoder_input,
            encoder_output=encoder_output,
            decoder_output=decoder_output,
        )
        return losses


class VAEBuilder:
    def __init__(
        self, cfg: VAEConfig, custom_loss: VAELoss, initialize: bool = True
    ) -> None:
        self.cfg = cfg
        self.custom_loss = custom_loss
        self.initialize = initialize

        encoder_config = EncoderConfig(
            input_shape=(cfg.original_dim,),
            input_shape_r=(cfg.original_dim_r,),
            dim_layer1=cfg.dim_layer1,
            dim_layer2=cfg.dim_layer2,
            dim_layer3=cfg.dim_layer3,
            activation=cfg.activation,
            cluster_number=cfg.cluster_number,
            latent_dim=cfg.latent_dim,
            pr_cluster_number=cfg.pr_cluster_number,
            sampling_fn=cfg.sampling_fn,
        )
        self.encoder = EncoderBuilder(encoder_config).build()

        decoder_config = DecoderConfig(
            dim_layer1=cfg.dim_layer1,
            dim_layer2=cfg.dim_layer2,
            dim_layer3=cfg.dim_layer3,
            activation=cfg.activation,
            latent_dim=cfg.latent_dim,
            output_dim=cfg.original_dim,
        )
        self.decoder = DecoderBuilder(decoder_config).build()

    def build(self) -> Model:
        vae = VAEModel(
            encoder=self.encoder,
            decoder=self.decoder,
            custom_loss=self.custom_loss,
            name="vae",
        )
        if self.initialize:
            self._initialize_model(model=vae)
        return vae

    def _initialize_model(self, model: Model) -> None:
        x_dummy = {
            "x": tf.zeros((1, self.cfg.original_dim)),
            "dummy": tf.zeros((1, 1)),
            "r": tf.zeros((1, self.cfg.pr_cluster_number)),
        }
        _ = model(x_dummy)


def _format_input_and_output(
    inputs: dict, outputs: dict
) -> tuple[EncoderInput, EncoderOutput, DecoderOutput]:
    """Format the encoder and decoder inputs and outputs as custom classes."""
    encoder_input = utils.format_encoder_input_as_object(encoder_input=inputs)
    encoder_output = utils.format_encoder_output_as_object(
        encoder_output=outputs["encoder_output"]
    )
    decoder_output = utils.format_decoder_output_as_object(
        decoder_output=outputs["decoder_output"]
    )
    return encoder_input, encoder_output, decoder_output


def _format_losses(total_loss: tf.Tensor, losses: Loss) -> dict[str, tf.Tensor]:
    """Format losses in order to retrieve the breakdown of the total loss."""
    return {
        "loss": total_loss,
        "reconstruction": tf.reduce_mean(losses.reconstruction),
        "kl_categorical": tf.reduce_mean(losses.kl_categorical),
        "dirichlet": tf.reduce_mean(losses.dirichlet),
        "target": tf.reduce_mean(losses.target),
        "kl_gaussian": tf.reduce_mean(losses.kl_gaussian),
    }
