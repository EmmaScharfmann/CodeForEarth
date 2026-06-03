import numpy as np
import tensorflow as tf

from cleaner_code_for_cmmvae.training import data_processor


class VAETrainer:
    def __init__(
        self,
        model: tf.keras.Model,
        batch_size: int,
    ) -> None:
        self.model = model
        self.data_module = data_processor.VAEDataModule(batch_size=batch_size)

    def load_initial_weights(self, path: str) -> None:
        """
        Load initial weights from the file stored at the given path.

        :param path:    The path the initial weights are stored at.
        """
        self.model.load_weights(path)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        filepath: str,
    ) -> tf.keras.callbacks.History:

        train_inputs, val_inputs = data_processor.train_val_split(X=X, y=y)

        train_ds = self.data_module.to_dataset(train_inputs)
        val_ds = self.data_module.to_dataset(val_inputs)

        self.load_initial_weights(path=filepath)

        history = self.model.fit(
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
        self.model.save_weights(path)
