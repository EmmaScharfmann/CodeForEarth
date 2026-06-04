import keras
from tensorflow.keras.layers import Input, Dense, Reshape, Lambda
from tensorflow.keras.models import Model

from package_name.training.utils import EncoderConfig


class EncoderBuilder:
    def __init__(self, config: EncoderConfig):
        self.cfg = config

    def build(self) -> Model:
        """Build the encoder model."""
        inputs = self._format_inputs()

        # based on x only (image to reconstruct - z500)
        x_reduced = self._reduce_vector_size(vector_input=inputs["x"])
        latent = self._build_latent_space(x_reduced=x_reduced)
        aux = self._build_aux_outputs(aux_input=x_reduced)

        # based on dummy only (fixed value of 1)
        mixture = self._build_mixture_components(dummy_input=inputs["dummy"])

        outputs = {"latent": latent, "mixture": mixture, "aux": aux}

        return Model(
            inputs=inputs,
            outputs=outputs,
            name="encoder",
        )

    def _format_inputs(self) -> dict[str, keras.KerasTensor]:
        """Format input as specified in the given config."""
        cfg = self.cfg
        x = Input(shape=cfg.input_shape, name="x")
        dummy = Input(shape=(1,), name="dummy")
        r = Input(shape=cfg.input_shape_r, name="r")

        return {
            "x": x,
            "dummy": dummy,
            "r": r,
        }

    def _reduce_vector_size(self, vector_input: keras.KerasTensor) -> keras.KerasTensor:
        """
        Encode the vector input to a vector of lower dimension.

        :param vector_input:    The vector to encode.
        :return:                The compact representation of the given vector.
        """
        cfg = self.cfg

        x = Dense(cfg.dim_layer1, activation=cfg.activation, name="enc_dense_1")(
            vector_input
        )
        x = Dense(cfg.dim_layer2, activation=cfg.activation, name="enc_dense_2")(x)
        x = Dense(cfg.dim_layer3, activation=cfg.activation, name="enc_dense_3")(x)

        return x

    def _build_latent_space(self, x_reduced: keras.KerasTensor):
        """
        Converts the given encoded vector into a probabilistic representation.

        :param x_reduced:    The vector to convert into a probabilistic representation.
        :return:             The latent space of the given vector.
        """
        cfg = self.cfg

        z_mean = Dense(cfg.latent_dim, name="z_mean")(x_reduced)
        z_log_var = Dense(cfg.latent_dim, name="z_log_var")(x_reduced)
        z = Lambda(cfg.sampling_fn, name="z")([z_mean, z_log_var])

        return {"z_mean": z_mean, "z_log_var": z_log_var, "z": z}

    def _build_mixture_components(self, dummy_input: keras.KerasTensor):
        """
        TODO
        """
        cfg = self.cfg

        mu_vector = Dense(
            cfg.cluster_number * cfg.latent_dim,
            use_bias=False,
            name="mu_vector",
        )(dummy_input)

        mu = Reshape(target_shape=(cfg.cluster_number, cfg.latent_dim), name="mu")(
            mu_vector
        )
        pi = Dense(cfg.cluster_number, activation="softmax", name="pi")(dummy_input)

        return {"mu": mu, "pi": pi}

    def _build_aux_outputs(self, aux_input: keras.KerasTensor):
        """
        TODO
        """
        cfg = self.cfg

        c = Dense(cfg.cluster_number, activation="softmax", name="c")(aux_input)
        r = Dense(cfg.pr_cluster_number, activation="softmax", name="r_label")(
            aux_input
        )
        cr = Dense(cfg.cluster_number, activation="softmax", name="cr")(r)

        return {"c": c, "r": r, "cr": cr}
