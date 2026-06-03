import numpy as np
import tensorflow as tf

from cleaner_code_for_cmmvae.training.trainer import VAETrainer


class CrossValidator:
    def __init__(
        self, trainer: VAETrainer, filepath: str, cluster_number: int, n_runs: int = 20
    ):
        self.trainer = trainer
        self.filepath = filepath
        self.cluster_number = cluster_number
        self.n_runs = n_runs

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
    ) -> list[tf.keras.callbacks.History]:
        """
        Run the cross validation for the given parameters and save the weights.

        :param X:       The input data the cross validation is run for.
        :param y:       The output data the cross validation is run for.
        :param epochs:  The number of epochs the cross validation is run for.
        :return:        The list of histories the cross validation is run for.
        """
        histories: list[tf.keras.callbacks.History] = []

        for i in range(self.n_runs):
            history = self._single_run(X, y, epochs, run_id=i)
            histories.append(history)

        return histories

    def _single_run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        run_id: int,
    ) -> tf.keras.callbacks.History:
        """Run the training for the given parameters and save the weights."""
        initial_weights_path = self._construct_initial_weights_path()
        final_weights_path = self._construct_final_weights_path(run_id)

        self.trainer.load_initial_weights(path=initial_weights_path)

        history = self.trainer.fit(
            X=X,
            y=y,
            epochs=epochs,
            filepath=initial_weights_path,
        )

        self.trainer.save_weights(path=final_weights_path)

        return history

    def _construct_initial_weights_path(self) -> str:
        """Construct the path where the initial weights are stored."""
        return f"{self.filepath}random_weights_{self.cluster_number}.h5"

    def _construct_final_weights_path(self, run_id: int) -> str:
        """Construct the path where the final weights are stored for the given run ID."""
        return f"{self.filepath}final_weights_{self.cluster_number}_{run_id}.h5"
