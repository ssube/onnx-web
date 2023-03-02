# this file exists to make sure torch is always imported before onnxruntime
# to work around https://github.com/microsoft/onnxruntime/issues/11092

import torch  # NOQA
from onnxruntime import *  # NOQA
