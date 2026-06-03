from tensorflow.keras import backend as K

from cleaner_code_for_cmmvae.training.model import (
    LatentSpace,
    MixtureComponents,
    AuxiliaryOutput,
    EncoderOutput,
    DecoderOutput,
    EncoderInput,
)


def format_encoder_output_as_object(encoder_output: dict) -> EncoderOutput:
    latent_input = LatentSpace(
        z=encoder_output["latent"]["z"],
        z_mean=encoder_output["latent"]["z_mean"],
        z_log_var=encoder_output["latent"]["z_log_var"],
    )

    mixture_input = MixtureComponents(
        mu=encoder_output["mixture"]["mu"],
        pi=encoder_output["mixture"]["pi"],
    )

    aux_input = AuxiliaryOutput(
        c=encoder_output["aux"]["c"],
        r=encoder_output["aux"]["r"],
        cr=encoder_output["aux"]["cr"],
    )

    encoder_output = EncoderOutput(
        latent=latent_input,
        mixture=mixture_input,
        aux=aux_input,
    )

    return encoder_output


def format_decoder_output_as_object(
    decoder_output: dict,
) -> DecoderOutput:
    decoder_output = DecoderOutput(x_recon=decoder_output["x_recon"])
    return decoder_output


def format_encoder_input_as_object(
    encoder_input: dict,
) -> EncoderInput:
    decoder_output = EncoderInput(
        x=encoder_input["x"], dummy=encoder_input["dummy"], r=encoder_input["r"]
    )
    return decoder_output


def sampling(args):
    """TODO"""
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(
        shape=(K.shape(z_mean)[0], K.int_shape(z_mean)[1]), mean=0.0, stddev=1.0
    )
    return z_mean + K.exp(0.5 * z_log_sigma) * epsilon
