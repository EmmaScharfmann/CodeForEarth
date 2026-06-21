import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from package_name.data_processing import data_processor
from package_name.model.decoder import DecoderBuilder
from package_name.model.encoder import EncoderBuilder
from package_name.model.loss import VAELoss
from package_name.model.vae_model import VAEModel
from package_name.model.utils import VAEConfig, EncoderConfig, DecoderConfig


class VAETrainer:
    def __init__(
        self,
        cfg: VAEConfig,
        reconstruction_loss_factor: float = 0.5,
        dirichlet_loss_factor: float = 0.5,
        path_to_save_weights: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.custom_loss = VAELoss(
            reconstruction_loss_factor=reconstruction_loss_factor,
            dirichlet_loss_factor=dirichlet_loss_factor,
            original_dim=cfg.original_dim,
            pr_cluster_number=cfg.pr_cluster_number,
        )
        self.path_to_save_weights = path_to_save_weights

        encoder_config = EncoderConfig(
            original_dim=cfg.original_dim,
            original_dim_target=cfg.original_dim_target,
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
            original_dim=cfg.original_dim,
        )
        self._decoder = DecoderBuilder(decoder_config).build()

        self._model = VAEModel(
            encoder=self._encoder,
            decoder=self._decoder,
            custom_loss=self.custom_loss,
            name="vae",
        )
        if self.path_to_save_weights is not None:
            self._initialize_and_save_weights(model=self._model)

    def initialize_weights_from(self, path: str) -> None:
        """
        Load initial weights from the file stored at the given path.

        :param path:    The path the initial weights are stored at.
        """
        self._model.load_weights(path)

    def compile(self) -> None:
        """Compile the model."""
        self._model.compile(optimizer="adam")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
    ) -> tf.keras.callbacks.History:
        """
        Fit the model to the given training data.

        :param X:           The input `X` to be fitted.
        :param y:           The output `y` to be fitted.
        :param epochs:      The number of epochs to train the model.
        :param batch_size:  The batch size to train the model.
        :return:            The history of the model.
        """
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

    def _initialize_and_save_weights(self, model: Model) -> None:
        """Build the model with a dummy forward pass and save the initial weights."""
        if self.path_to_save_weights is None:
            return

        self._build()
        model.save_weights(
            os.path.join(
                self.path_to_save_weights,
                f"random_weights_{str(self.cfg.cluster_number)}.weights.h5",
            )
        )

    def _build(self) -> None:
        """Trigger a forward pass to initialize weight shapes."""
        dummy_input = {
            "x": tf.zeros((1, self.cfg.original_dim)),
            "dummy": tf.zeros((1, 1)),
            "target": tf.zeros((1, self.cfg.pr_cluster_number)),
        }
        self._model(dummy_input)
