from typing import override

from tensorflow.keras.models import Model
import tensorflow as tf

from package_name.training import utils
from package_name.training.loss import VAELoss


class VAEModel(Model):
    def __init__(self, encoder: Model, decoder: Model, custom_loss: VAELoss, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.custom_loss = custom_loss

    @override
    def call(self, encoder_input: dict, training: bool = False, **kwargs):
        """
        Get the encoder output and decoder output from the encoder input.

        :param encoder_input:   The encoder input.
        :param training:        Whether the model is in training mode.
        :param kwargs:          Other arguments, to match the parent function `call`.
        :return:                A dictionary with the encoder output and decoder output.
        """
        encoder_output = self.encoder(encoder_input)
        z = encoder_output["latent"]["z"]
        decoder_output = self.decoder({"z": z})

        losses = self.custom_loss.compute(
            encoder_input=utils.format_encoder_input_as_object(
                encoder_input=encoder_input
            ),
            encoder_output=utils.format_encoder_output_as_object(
                encoder_output=encoder_output
            ),
            decoder_output=utils.format_decoder_output_as_object(
                decoder_output=decoder_output
            ),
        )
        self.add_loss(tf.reduce_mean(losses.total))
        return {
            "encoder_output": {
                "latent": {
                    "z": z,
                    "z_mean": encoder_output["latent"]["z_mean"],
                    "z_log_var": encoder_output["latent"]["z_log_var"],
                },
                "mixture": encoder_output["mixture"],
                "aux": encoder_output["aux"],
            },
            "decoder_output": {"x_recon": decoder_output["x_recon"]},
        }
