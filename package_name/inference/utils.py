from tensorflow.keras.models import Model


def build_inference_encoder(full_encoder: Model) -> Model:
    """
    Extracts an inference-only model from a full encoder model.
    It strips out the "dummy" and "target" inputs, and only outputs "z" and
    "clusters_pred".

    :param full_encoder: The full encoder model, which takes 'x', 'dummy', and 'target'
        as inputs and outputs "latent", "mixture" and "clusters" dictionaries.
    :return: A new Keras Model that takes only "x" as input and outputs "z" and
        "clusters_pred"
    """
    inference_input = full_encoder.get_layer("x").output

    inference_outputs = {
        "z": full_encoder.get_layer("z").output,
        "clusters_pred": full_encoder.get_layer("clusters_pred").output,
    }

    return Model(
        inputs=inference_input, outputs=inference_outputs, name="inference_encoder"
    )
