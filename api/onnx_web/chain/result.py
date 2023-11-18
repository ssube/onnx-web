from PIL.Image import Image, fromarray
from typing import List, Optional

import numpy as np

class StageResult:
  """
  Chain pipeline stage result.
  Can contain PIL images or numpy arrays, with helpers to convert between them.
  """
  arrays: Optional[List[np.ndarray]]
  images: Optional[List[Image]]

  def __init__(self, arrays = None, images = None) -> None:
    if arrays is not None and images is not None:
      raise ValueError("stages must only return one type of result")

    self.arrays = arrays
    self.images = images

  def as_numpy(self) -> List[np.ndarray]:
    if self.arrays is not None:
      return self.arrays

    return [np.array(i) for i in self.images]

  def as_image(self) -> List[Image]:
    if self.images is not None:
      return self.images

    return [fromarray(i) for i in self.arrays]
