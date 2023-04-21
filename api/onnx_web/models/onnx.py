from logging import getLogger
from os import path
from typing import Any, Optional

from ..server import ServerContext
from ..torch_before_ort import InferenceSession, SessionOptions

logger = getLogger(__name__)


class OnnxModel:
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
        output = self.session.run([output_name], {input_name: image})[0]
        return output
