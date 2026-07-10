import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from package_name.data_processing import data_processor
from package_name.model.decoder import DecoderBuilder
from package_name.model.encoder import EncoderBuilder
from package_name.model.loss import VAELoss
from package_name.model.vae_model import VAEModel
from package_name.model.utils import (
    VAEConfig,
    construct_encoder_config,
    construct_decoder_config,
    LossFactorsConfig,
)


class VAETrainer:
    def __init__(
        self,
        cfg: VAEConfig,
        loss_factors: LossFactorsConfig,
        path_to_save_weights: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.custom_loss = VAELoss(
            original_dim=cfg.original_dim,
            pr_cluster_number=cfg.pr_cluster_number,
            loss_factors=loss_factors
        )
        self.path_to_save_weights = path_to_save_weights

        self._encoder = EncoderBuilder(construct_encoder_config(cfg=self.cfg), training=True).build()
        self._decoder = DecoderBuilder(construct_decoder_config(cfg=self.cfg)).build()
        self._model = VAEModel(
            encoder=self._encoder,
            decoder=self._decoder,
            custom_loss=self.custom_loss,
            name="vae",
        )
        self._build_and_save_weights(model=self._model)

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

    def _build_and_save_weights(self, model: Model) -> None:
        """Build the model with a dummy forward pass and save the initial weights."""
        self._model.build()
        if self.path_to_save_weights is None:
            return

        model.save_weights(
            os.path.join(
                self.path_to_save_weights,
                f"random_weights_{str(self.cfg.cluster_number)}.weights.h5",
            )
        )
