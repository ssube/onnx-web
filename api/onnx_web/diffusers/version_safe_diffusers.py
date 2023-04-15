import diffusers
from diffusers import *  # NOQA
from packaging import version

is_diffusers_0_15 = version.parse(
    version.parse(diffusers.__version__).base_version
) >= version.parse("0.15")


try:
    from diffusers import DEISMultistepScheduler  # NOQA
except ImportError:
    from ..diffusers.stub_scheduler import (
        StubScheduler as DEISMultistepScheduler,  # NOQA
    )

try:
    from diffusers import UniPCMultistepScheduler  # NOQA
except ImportError:
    from ..diffusers.stub_scheduler import (
        StubScheduler as UniPCMultistepScheduler,  # NOQA
    )


if is_diffusers_0_15:
    from diffusers.models.attention_processor import AttnProcessor  # NOQA
else:
    from diffusers.models.cross_attention import (
        CrossAttnProcessor as AttnProcessor,  # NOQA
    )
