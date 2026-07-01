import numpy as np
import tensorflow as tf

from package_name.model.decoder import DecoderBuilder
from package_name.model.encoder import EncoderBuilder
from package_name.model.loss import VAELoss
from package_name.model.vae_model import VAEModel
from package_name.model.utils import VAEConfig, EncoderConfig, DecoderConfig


class VAEPredictor:
    def __init__(
        self,
        cfg: VAEConfig,
        reconstruction_loss_factor: float = 0.5,
        dirichlet_loss_factor: float = 0.5,
        regularisation_loss_factor: float = 0.5,
        target_prediction_loss_factor: float = 0.5,
        cluster_target_regularisation_loss_factor: float = 0.5,
        mixture_regularization_loss_factor: float = 0.5,
    ) -> None:
        self.cfg = cfg
        self.custom_loss = VAELoss(
            reconstruction_loss_factor=reconstruction_loss_factor,
            dirichlet_loss_factor=dirichlet_loss_factor,
            original_dim=cfg.original_dim,
            pr_cluster_number=cfg.pr_cluster_number,
            regularisation_loss_factor=regularisation_loss_factor,
            target_prediction_loss_factor=target_prediction_loss_factor,
            cluster_target_regularisation_loss_factor=cluster_target_regularisation_loss_factor,
            mixture_regularization_loss_factor=mixture_regularization_loss_factor,
        )

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

    # TODO: change the encoder structure to only have to pass X instead of having to pass dummies "dummy" and "target" alongside.
    def encode(self, input: dict[str,np.ndarray], batch_size: int) -> np.ndarray:
        """
        Encode the given input `X`.

        :param input:       The encoder input with the following format:
                            {"x": X,
                             "dummy": np.ones((X.shape[0], 1)),
                             "target": np.zeros((X.shape[0], vae.cfg.pr_cluster_number)),
                            }
                            where X is the matrix to encode.
        :param batch_size:  The number of samples per batch of computation
        :return:            The encoded `X`.
        """
        return self._encoder.predict(x=input, batch_size=batch_size)

    def _build(self) -> None:
        """Trigger a forward pass to initialize weight shapes."""
        dummy_input = {
            "x": tf.zeros((1, self.cfg.original_dim)),
            "dummy": tf.zeros((1, 1)),
            "target": tf.zeros((1, self.cfg.pr_cluster_number)),
        }
        self._model(dummy_input)

    def load_weights(self, path: str) -> None:
        """
        Load the model weights from the given path.

        :param path:    The path the model weights are stored at.
        """
        self._build()
        self._model.load_weights(path)