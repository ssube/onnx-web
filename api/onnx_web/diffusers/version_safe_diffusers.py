import diffusers
from diffusers import *  # NOQA
from packaging import version

is_diffusers_0_15 = version.parse(
    version.parse(diffusers.__version__).base_version
) >= version.parse("0.15")
is_diffusers_0_24 = version.parse(
    version.parse(diffusers.__version__).base_version
) >= version.parse("0.24")

try:
    from diffusers import DEISMultistepScheduler
except ImportError:
    from ..diffusers.stub_scheduler import StubScheduler as DEISMultistepScheduler

try:
    from diffusers import DPMSolverSDEScheduler
except ImportError:
    from ..diffusers.stub_scheduler import StubScheduler as DPMSolverSDEScheduler

try:
    from diffusers import LCMScheduler
except ImportError:
    from ..diffusers.stub_scheduler import StubScheduler as LCMScheduler

try:
    from diffusers import UniPCMultistepScheduler
except ImportError:
    from ..diffusers.stub_scheduler import StubScheduler as UniPCMultistepScheduler


if is_diffusers_0_15:
    from diffusers.models.attention_processor import AttnProcessor
else:
    from diffusers.models.cross_attention import CrossAttnProcessor as AttnProcessor


if is_diffusers_0_24:
    from diffusers.models.modeling_outputs import AutoencoderKLOutput
    from diffusers.models.autoencoders.vae import DecoderOutput
else:
    from diffusers.models.autoencoder_kl import AutoencoderKLOutput
    from diffusers.models.vae import DecoderOutput