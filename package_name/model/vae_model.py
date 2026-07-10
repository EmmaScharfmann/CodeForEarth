from typing import override

from tensorflow.keras.models import Model
import tensorflow as tf

from package_name.model import utils
from package_name.model.loss import VAELoss


class VAEModel(Model):
    def __init__(
        self,
        encoder: Model,
        decoder: Model,
        custom_loss: VAELoss | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.custom_loss = custom_loss
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.vae_regularisation_loss_tracker = tf.keras.metrics.Mean(name="vae_regularisation_loss")
        self.target_prediction_loss_tracker = tf.keras.metrics.Mean(name="target_prediction_loss")
        self.cluster_target_regularisation_loss_tracker = tf.keras.metrics.Mean(name="cluster_target_regularisation_loss")
        self.mixture_regularization_loss_tracker = tf.keras.metrics.Mean(name="mixture_regularization_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        
        self.tracker_mapping = {
            "vae_reconstruction": self.reconstruction_loss_tracker,
            "vae_regularisation": self.vae_regularisation_loss_tracker,
            "target_prediction": self.target_prediction_loss_tracker,
            "cluster_target_regularisation": self.cluster_target_regularisation_loss_tracker,
            "mixture_regularization": self.mixture_regularization_loss_tracker,
            "total": self.total_loss_tracker,}



    @override
    def call(self, encoder_input: dict, **kwargs):
        """
        Get the encoder output and decoder output from the encoder input.

        :param encoder_input:   The encoder input.
        :param kwargs:          Other arguments, to match the parent function `call`.
        :return:                A dictionary with the encoder output and decoder output.
        """
        if self.custom_loss is None:
            raise ValueError("A custom loss is necessary for training.")

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
        self.total_loss_tracker.update_state(losses.total)

        for loss_key, tensor_value in losses.individual_losses().items():
            self.tracker_mapping[loss_key].update_state(tensor_value)

            
        return {
            "encoder_output": {
                "latent": {
                    "z": z,
                    "z_mean": encoder_output["latent"]["z_mean"],
                    "z_log_var": encoder_output["latent"]["z_log_var"],
                },
                "mixture": encoder_output["mixture"],
                "clusters": encoder_output["clusters"],
            },
            "decoder_output": {"x_recon": decoder_output["x_recon"]},
        }

    @property
    def metrics(self):
        """
        Returns list of metrics for the model.
        """
        return [
            self.reconstruction_loss_tracker,
            self.vae_regularisation_loss_tracker,
            self.target_prediction_loss_tracker,
            self.cluster_target_regularisation_loss_tracker,
            self.mixture_regularization_loss_tracker,
            self.total_loss_tracker,
        ]
    def build(self, input_shape=None):
        super().build(input_shape)
