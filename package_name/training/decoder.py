import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

from package_name.training.utils import DecoderConfig


class DecoderBuilder:
    def __init__(self, config: DecoderConfig):
        self.cfg = config

    def build(self) -> Model:
        """Build the decoder model."""
        z = self._create_latent_input()
        outputs = self._build_decoder_network(encoded_vector=z)

        return Model(inputs={"z": z}, outputs={"x_recon": outputs}, name="decoder")

    def _create_latent_input(self) -> keras.KerasTensor:
        """Create the latent input from the given config."""
        return Input(shape=(self.cfg.latent_dim,), name="z")

    def _build_decoder_network(self, encoded_vector:  keras.KerasTensor) -> keras.KerasTensor:
        """
        Decode the given vector to a vector of higher dimension.

        :param encoded_vector:    The vector to encode.
        :return:                  The decoded representation of the given vector.
        """
        cfg = self.cfg

        x = Dense(cfg.dim_layer3, activation=cfg.activation, name="dec_dense_1")(
            encoded_vector
        )
        x = Dense(cfg.dim_layer2, activation=cfg.activation, name="dec_dense_2")(x)
        x = Dense(cfg.dim_layer1, activation=cfg.activation, name="dec_dense_3")(x)
        x = Dense(self.cfg.output_dim, name="x_recon")(x)
        return x
