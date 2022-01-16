from typing import Union

from models.vae import BaseVAE, VAEHalfInputSize, VAEFullInputSize, LargeVAESmallKernels
from models.rnn import BaseRNN, StandardMDNRNN, SimpleLSTM

vae_models = {
    "vae_half_input_size": VAEHalfInputSize,
    "vae_full_input_size": VAEFullInputSize,
    "large_vae_small_kernel": LargeVAESmallKernels
}

rnn_models = {
    "standard_mdn": StandardMDNRNN,
    "lstm": SimpleLSTM
}


def _select_model(model_name: str, model_type: str) -> Union[BaseVAE, BaseRNN]:
    if model_type == "VAE":
        available_models = vae_models
    else:
        available_models = rnn_models

    try:
        selected_model = available_models[model_name]
    except KeyError:
        raise RuntimeError(f"{model_type} architecture '{model_name}' does not exist, available architectures are "
                           "listed in models/model_selection.py")

    return selected_model


def select_vae_model(model_name: str) -> BaseVAE:
    return _select_model(model_name, "VAE")


def select_rnn_model(model_name: str) -> BaseRNN:
    return _select_model(model_name, "RNN")
