import keras
from tensorflow.keras.layers import Input, Dense, Reshape, Lambda
from tensorflow.keras.models import Model

from package_name.training.utils import EncoderConfig


class EncoderBuilder:
    def __init__(self, config: EncoderConfig):
        self.cfg = config

    def build(self) -> Model:
        """
        Build the encoder model.

        :return:        The encoder model.
        """
        inputs = self._format_inputs()

        x_reduced = self._reduce_vector_size(vector_input=inputs["x"])
        latent = self._build_latent_space(x_reduced=x_reduced)
        clusters = self._build_cluster_outputs(clusters_input=x_reduced)

        mixture = self._build_mixture_components(dummy_input=inputs["dummy"])

        outputs = {"latent": latent, "mixture": mixture, "clusters": clusters}

        return Model(
            inputs=inputs,
            outputs=outputs,
            name="encoder",
        )

    def _format_inputs(self) -> dict[str, keras.KerasTensor]:
        """Format input as specified in the given config."""
        cfg = self.cfg
        x = Input(shape=(cfg.original_dim,), name="x")
        dummy = Input(shape=(1,), name="dummy")
        target = Input(shape=(cfg.original_dim_target,), name="target")

        return {
            "x": x,
            "dummy": dummy,
            "target": target,
        }

    def _reduce_vector_size(self, vector_input: keras.KerasTensor) -> keras.KerasTensor:
        """Encode the vector input to a vector of lower dimension."""
        cfg = self.cfg

        x = Dense(cfg.dim_layer1, activation=cfg.activation, name="enc_dense_1")(
            vector_input
        )
        x = Dense(cfg.dim_layer2, activation=cfg.activation, name="enc_dense_2")(x)
        x = Dense(cfg.dim_layer3, activation=cfg.activation, name="enc_dense_3")(x)

        return x

    def _build_latent_space(self, x_reduced: keras.KerasTensor):
        """Converts the given encoded vector into a probabilistic representation."""
        cfg = self.cfg

        z_mean = Dense(cfg.latent_dim, name="z_mean")(x_reduced)
        z_log_var = Dense(cfg.latent_dim, name="z_log_var")(x_reduced)
        z = Lambda(cfg.sampling_fn, name="z")([z_mean, z_log_var])

        return {"z_mean": z_mean, "z_log_var": z_log_var, "z": z}

    def _build_mixture_components(
        self, dummy_input: keras.KerasTensor
    ) -> dict[str, keras.KerasTensor]:
        """Build the mixture model outputs (cluster means and mixing probabilities) from the input tensor."""
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

    def _build_cluster_outputs(
        self, clusters_input: keras.KerasTensor
    ) -> dict[str, keras.KerasTensor]:
        """Build the cluster outputs, either directly from the input variable (cluster_pred) or from the target variable (target_cluster_pred)."""
        cfg = self.cfg

        c = Dense(cfg.cluster_number, activation="softmax", name="c")(clusters_input)
        target_pred = Dense(
            cfg.pr_cluster_number, activation="softmax", name="target_pred"
        )(clusters_input)

        clusters_pred_from_target = Dense(
            cfg.cluster_number, activation="softmax", name="cr"
        )(target_pred)

        return {
            "clusters_pred": c,
            "target_pred": target_pred,
            "target_clusters_pred": clusters_pred_from_target,
        }
