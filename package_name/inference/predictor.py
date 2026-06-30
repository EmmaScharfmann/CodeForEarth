import numpy as np

from package_name.model.decoder import DecoderBuilder
from package_name.model.encoder import EncoderBuilder
from package_name.model.vae_model import VAEModel
from package_name.model.utils import (
    VAEConfig,
    construct_decoder_config,
    construct_encoder_config,
)


class VAEPredictor:
    def __init__(
        self,
        cfg: VAEConfig,
    ) -> None:
        self.cfg = cfg
        self._encoder = EncoderBuilder(construct_encoder_config(cfg=self.cfg)).build()
        self._decoder = DecoderBuilder(construct_decoder_config(cfg=self.cfg)).build()
        self._model = VAEModel(
            encoder=self._encoder,
            decoder=self._decoder,
            name="vae",
        )

    # TODO: change the encoder structure to only have to pass X instead of having to pass dummies "dummy" and "target" alongside.
    def encode(self, input: dict[str, np.ndarray], batch_size: int) -> np.ndarray:
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

    def decode(self, z: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Decode the given input `z`.

        :param z:           The decoder input (corresponding to `z` in the LatentSpaceOutput) with the following format: (self.cfg.latent_dim,)
        :param batch_size:  The number of samples per batch of computation
        :return:            The decoded input 'z'.
        """
        return self._decoder.predict(x={"z": z}, batch_size=batch_size)

    def load_weights(self, path: str) -> None:
        """
        Load the model weights from the given path.

        :param path:    The path the model weights are stored at.
        """
        self._model.build()
        self._model.load_weights(path)
