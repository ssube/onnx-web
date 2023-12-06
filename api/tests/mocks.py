from typing import Any, Optional


class MockPipeline:
    # flags
    slice_size: Optional[str]
    vae_slicing: Optional[bool]
    sequential_offload: Optional[bool]
    model_offload: Optional[bool]
    xformers: Optional[bool]

    # stubs
    _encode_prompt: Optional[Any]
    unet: Optional[Any]
    vae_decoder: Optional[Any]
    vae_encoder: Optional[Any]

    def __init__(self) -> None:
        self.slice_size = None
        self.vae_slicing = None
        self.sequential_offload = None
        self.model_offload = None
        self.xformers = None

        self._encode_prompt = None
        self.unet = None
        self.vae_decoder = None
        self.vae_encoder = None

    def enable_attention_slicing(self, slice_size: str = None):
        self.slice_size = slice_size

    def enable_vae_slicing(self):
        self.vae_slicing = True

    def enable_sequential_cpu_offload(self):
        self.sequential_offload = True

    def enable_model_cpu_offload(self):
        self.model_offload = True

    def enable_xformers_memory_efficient_attention(self):
        self.xformers = True
