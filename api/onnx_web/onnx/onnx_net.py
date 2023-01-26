from onnxruntime import InferenceSession
from os import path
from typing import Any

import numpy as np
import torch

from ..utils import (
  ServerContext,
)

class ONNXImage():
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


class ONNXNet():
    '''
    Provides the RRDBNet interface using an ONNX session for DirectML acceleration.
    '''

    def __init__(self, ctx: ServerContext, model: str, provider='DmlExecutionProvider') -> None:
        '''
        TODO: get platform provider from request params
        '''
        model_path = path.join(ctx.model_path, model)
        self.session = InferenceSession(
            model_path, providers=[provider])

    def __call__(self, image: Any) -> Any:
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        output = self.session.run([output_name], {
            input_name: image.cpu().numpy()
        })[0]
        return ONNXImage(output)

    def eval(self) -> None:
        pass

    def half(self):
        return self

    def load_state_dict(self, net, strict=True) -> None:
        pass

    def to(self, device):
        return self
