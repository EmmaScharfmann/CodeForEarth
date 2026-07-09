import numpy as np

from package_name.model.decoder import DecoderBuilder
from package_name.model.encoder import EncoderBuilder
from package_name.model.vae_model import VAEModel
from package_name.model.utils import (
    VAEConfig,
    construct_decoder_config,
    construct_encoder_config,
)
from package_name.inference.utils import build_inference_encoder


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
        self._inference_encoder = build_inference_encoder(full_encoder=self._encoder)

    def encode(self, input: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        """
        Encode the given input `X`.

        :param input:       The encoder input 'X'
        :param batch_size:  The number of samples per batch of computation
        :return:            The encoded `X`.
        """
        return self._inference_encoder.predict(x=input, batch_size=batch_size)

    def decode(self, z: np.ndarray, batch_size: int | None = None) -> np.ndarray:
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

    # TODO: Find a cleaner way to get the mixture components (mu and pi) from the encoder, without relying on a dummy input.
    def get_mixture_components(self) -> dict[str, np.ndarray]:
        """
        Get the mixture components (cluster centers mu and probabilities pi) of the model.
        This is done by passing a dummy input through the encoder and extracting the
        mixture components from the output.

        :return:    A dictionary with the mixture components "mu" and "pi".
        """
        dummy_input = {
            "x": np.zeros((1, self.cfg.original_dim)),
            "dummy": np.ones((1, 1)),
            "target": np.zeros((1, self.cfg.pr_cluster_number)),
        }
        output = self._encoder.predict(x=dummy_input)
        return {
            "mu": output["mixture"]["mu"],
            "pi": output["mixture"]["pi"],
        }
