from models.vae.base_vae import BaseVAE
from models.vae.vae_half_input import VAEHalfInputSize
from models.vae.vae_full_input import VAEFullInputSize
from models.vae.large_vae_small_kernel import LargeVAESmallKernels, EvenLargerVAESmallKernels

from models.vae.no_max_pool import (
    LargeFilterSizesVAE, SmallFilterSizesVAE, DecreasingFilterSizesVAE, SmallFilterSizesSmallBottleneckVAE)
from models.vae.with_max_pool import (
    LargeFilterSizesMaxPoolVAE, SmallFilterSizesMaxPoolVAE, DecreasingFilterSizesMaxPoolVAE)
