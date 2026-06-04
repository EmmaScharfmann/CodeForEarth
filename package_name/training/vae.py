import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from package_name.data_processing import data_processor
from package_name.training import utils
from package_name.training.decoder import DecoderBuilder
from package_name.training.encoder import EncoderBuilder
from package_name.training.loss import VAELoss
from package_name.training.model import VAEModel
from package_name.training.utils import VAEConfig, EncoderConfig, DecoderConfig


class VAE:
    def __init__(
        self,
        cfg: VAEConfig,
        reconstruction_loss_factor: float = 0.5,
        path_for_weights_initialization: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.custom_loss = VAELoss(
            reconstruction_loss_factor=reconstruction_loss_factor,
            original_dim=cfg.original_dim,
            pr_cluster_number=cfg.pr_cluster_number,
        )
        self.path_for_weights_initialization = path_for_weights_initialization

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
        self._encoder = EncoderBuilder(encoder_config).build()

        decoder_config = DecoderConfig(
            dim_layer1=cfg.dim_layer1,
            dim_layer2=cfg.dim_layer2,
            dim_layer3=cfg.dim_layer3,
            activation=cfg.activation,
            latent_dim=cfg.latent_dim,
            output_dim=cfg.original_dim,
        )
        self._decoder = DecoderBuilder(decoder_config).build()

        self._model = VAEModel(
            encoder=self._encoder,
            decoder=self._decoder,
            custom_loss=self.custom_loss,
            name="vae",
        )
        if self.path_for_weights_initialization is not None:
            self._initialize_weights(model=self._model)
            self._model.save_weights(
                self.path_for_weights_initialization
                + "random_weights_"
                + str(cfg.cluster_number)
                + ".weights.h5"
            )

    def initialize_weights_from(self, path: str) -> None:
        """
        Load initial weights from the file stored at the given path.

        :param path:    The path the initial weights are stored at.
        """
        self._model.load_weights(path)

    def compile(self):
        """Compile the model."""
        self._model.compile(optimizer="adam")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
    ) -> tf.keras.callbacks.History:
        """TODO"""
        train_inputs, val_inputs = data_processor.train_val_split(X=X, y=y)

        train_ds = data_processor.format_input_to_dataset(
            inputs=train_inputs, batch_size=batch_size
        )
        val_ds = data_processor.format_input_to_dataset(
            inputs=val_inputs, batch_size=batch_size
        )

        history = self._model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
        )

        return history

    def save_weights(self, path: str) -> None:
        """
        Save the model weights to the given path.

        :param path:    The path the model weights are stored at.
        """
        self._model.save_weights(path)

    def _initialize_weights(self, model: Model) -> None:
        """Initialize the weights of the model."""
        x_dummy = {
            "x": tf.zeros((1, self.cfg.original_dim)),
            "dummy": tf.zeros((1, 1)),
            "r": tf.zeros((1, self.cfg.pr_cluster_number)),
        }
        _ = model(x_dummy)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode the given input `X`.

        :param X:   The input `X` to be encoded.
        :return:    The encoded `X`.
        """
        return self._encoder.predict(X)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """
        Decode the given output `Z`.

        :param Z:   The output of the encoder.
        :returns:   The decoded output.
        """
        return self._decoder.predict(Z)
