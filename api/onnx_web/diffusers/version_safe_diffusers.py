from diffusers import *  # NOQA

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

try:
    from diffusers.models.modeling_outputs import AutoencoderKLOutput
except ImportError:
    from diffusers.models.autoencoder_kl import AutoencoderKLOutput

try:
    from diffusers.models.autoencoders.vae import DecoderOutput
except ImportError:
    from diffusers.models.vae import DecoderOutput

try:
    from diffusers.models.attention_processor import AttnProcessor
except ImportError:
    from diffusers.models.cross_attention import CrossAttnProcessor as AttnProcessor
