import torch

class ConversionContext:
  def __init__(self, model_path: str, device: str) -> None:
    self.model_path = model_path
    self.training_device = device
    self.map_location = torch.device(device)