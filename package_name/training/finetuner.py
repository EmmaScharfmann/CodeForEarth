from package_name.training.trainer import VAETrainer
from package_name.model.utils import VAEConfig, LossFactorsConfig

LEARNING_RATE: float = 1e-4


class VAEFinetuner(VAETrainer):
    """
    A class for fine-tuning a pre-trained Variational Autoencoder (VAE) model.
    """
    def __init__(
        self,
        cfg: VAEConfig,
        loss_factors: LossFactorsConfig,
        path_to_load_weights: str | None = None,
        path_to_save_weights: str | None = None,
        learning_rate: float = LEARNING_RATE,
    ) -> None:
        super().__init__(
            cfg=cfg,
            loss_factors=loss_factors,
            path_to_save_weights=path_to_save_weights,
        )

        if path_to_load_weights is not None:
            self.initialize_weights_from(path_to_load_weights)

        # Freeze the encoder and decoder layers to prevent them from being updated during fine-tuning
        for layer_name in ["enc_dense_1", "enc_dense_2"]:
            self._encoder.get_layer(layer_name).trainable = False

        self._decoder.trainable = False

        self.compile(learning_rate=learning_rate)
