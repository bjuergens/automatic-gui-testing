from typing import Union

from models.vae import BaseVAE, VAEHalfInputSize, VAEFullInputSize, LargeVAESmallKernels, EvenLargerVAESmallKernels
from models.vae import (
    LargeFilterSizesVAE, SmallFilterSizesVAE, DecreasingFilterSizesVAE, SmallFilterSizesSmallBottleneckVAE,
    SmallFilterSizesSmallBottleneckVAE2, SmallFilterSizesSmallBottleneckVAE3,
    HalfInputSmallFilterSizesSmallBottleneckVAE, HalfInputSmallFilterSizesSmallBottleneckVAE2
)
from models.vae import (
    LargeFilterSizesMaxPoolVAE, SmallFilterSizesMaxPoolVAE, DecreasingFilterSizesMaxPoolVAE,
    SmallFilterSizesSmallBottleneckMaxPoolVAE, SmallFilterSizesSmallBottleneckMaxPoolVAE2,
    HalfInputSmallFilterSizesSmallBottleneckMaxPoolVAE, HalfInputSmallFilterSizesWithStrideMaxPoolVAE
)
from models.rnn import BaseRNN, StandardMDNRNN, LSTMWithBCE, LSTMWithMSE

vae_models = {
    "vae_half_input_size": VAEHalfInputSize,
    "vae_full_input_size": VAEFullInputSize,
    "large_vae_small_kernel": LargeVAESmallKernels,
    "even_larger_vae_small_kernel": EvenLargerVAESmallKernels,
    "large_filter_sizes": LargeFilterSizesVAE,
    "small_filter_sizes": SmallFilterSizesVAE,
    "small_filter_sizes_small_bottleneck": SmallFilterSizesSmallBottleneckVAE,
    "small_filter_sizes_small_bottleneck_2": SmallFilterSizesSmallBottleneckVAE2,
    "small_filter_sizes_small_bottleneck_3": SmallFilterSizesSmallBottleneckVAE3,
    "decreasing_filter_sizes": DecreasingFilterSizesVAE,
    "large_filter_sizes_maxpool": LargeFilterSizesMaxPoolVAE,
    "small_filter_sizes_maxpool": SmallFilterSizesMaxPoolVAE,
    "small_filter_sizes_small_bottleneck_maxpool": SmallFilterSizesSmallBottleneckMaxPoolVAE,
    "small_filter_sizes_small_bottleneck_maxpool_2": SmallFilterSizesSmallBottleneckMaxPoolVAE2,
    "decreasing_filter_sizes_maxpool": DecreasingFilterSizesMaxPoolVAE,
    "half_input_small_filter_sizes_small_bottleneck": HalfInputSmallFilterSizesSmallBottleneckVAE,
    "half_input_small_filter_sizes_small_bottleneck_2": HalfInputSmallFilterSizesSmallBottleneckVAE2,
    "half_input_small_filter_sizes_small_bottleneck_maxpool": HalfInputSmallFilterSizesSmallBottleneckMaxPoolVAE,
    "half_input_small_filter_sizes_with_stride_maxpool": HalfInputSmallFilterSizesWithStrideMaxPoolVAE,
}

rnn_models = {
    "standard_mdn": StandardMDNRNN,
    "lstm": LSTMWithBCE,  # For backwards compatibility
    "lstm_bce": LSTMWithBCE,
    "lstm_mse": LSTMWithMSE
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
