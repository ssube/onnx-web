from os import path
from typing import Any, Optional

import numpy as np
import torch
from onnxruntime import InferenceSession, SessionOptions

from ..utils import ServerContext


class OnnxImage:
    def __init__(self, source) -> None:
        self.source = source
        self.data = self

    def __getitem__(self, *args):
        return torch.from_numpy(self.source.__getitem__(*args)).to(torch.float32)

    def squeeze(self):
        self.source = np.squeeze(self.source, (0))
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def clamp_(self, min, max):
        self.source = np.clip(self.source, min, max)
        return self

    def numpy(self):
        return self.source

    def size(self):
        return np.shape(self.source)


class OnnxNet:
    """
    Provides the RRDBNet interface using an ONNX session for DirectML acceleration.
    """

    def __init__(
        self,
        server: ServerContext,
        model: str,
        provider: str = "DmlExecutionProvider",
        sess_options: Optional[SessionOptions] = None,
    ) -> None:
        model_path = path.join(server.model_path, model)
        self.session = InferenceSession(
            model_path, providers=[provider], provider_options=sess_options
        )

    def __call__(self, image: Any) -> Any:
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        output = self.session.run([output_name], {input_name: image.cpu().numpy()})[0]
        return OnnxImage(output)

    def eval(self) -> None:
        pass

    def half(self):
        return self

    def load_state_dict(self, net, strict=True) -> None:
        pass

    def to(self, device):
        return self
