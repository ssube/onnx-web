from collections import Counter
from logging import getLogger
from onnx import load_model
from os import path
from safetensors import safe_open
from sys import argv
from typing import Callable, List

import onnx_web
import torch

logger = getLogger(__name__)

CheckFunc = Callable[[str], List[str]]

def check_file_extension(filename: str) -> List[str]:
  """
  Check the file extension
  """
  _name, ext = path.splitext(filename)
  ext = ext.removeprefix(".")

  if ext != "":
    return [f"format:{ext}"]

  return []

def check_file_diffusers(filename: str) -> List[str]:
  """
  Check for a diffusers directory with model_index.json
  """
  if path.isdir(filename) and path.exists(path.join(filename, "model_index.json")):
    return ["model:diffusion"]

  return []

def check_parser_safetensor(filename: str) -> List[str]:
  """
  Attempt to load as a safetensor
  """
  try:
    if path.isfile(filename):
      # only try to parse files
      safe_open(filename, framework="pt")
      return ["format:safetensor"]
  except Exception as e:
    logger.debug("error parsing as safetensor: %s", e)

  return []

def check_parser_torch(filename: str) -> List[str]:
  """
  Attempt to load as a torch tensor
  """
  try:
    if path.isfile(filename):
      # only try to parse files
      torch.load(filename)
      return ["format:torch"]
  except Exception as e:
    logger.debug("error parsing as torch tensor: %s", e)

  return []

def check_parser_onnx(filename: str) -> List[str]:
  """
  Attempt to load as an ONNX model
  """
  try:
    if path.isfile(filename):
      load_model(filename)
      return ["format:onnx"]
  except Exception as e:
    logger.debug("error parsing as ONNX model: %s", e)

  return []

def check_network_lora(filename: str) -> List[str]:
  """
  TODO: Check for LoRA keys
  """
  return []

def check_network_inversion(filename: str) -> List[str]:
  """
  TODO: Check for Textual Inversion keys
  """
  return []


ALL_CHECKS: List[CheckFunc] = [
  check_file_diffusers,
  check_file_extension,
  check_network_inversion,
  check_network_lora,
  check_parser_onnx,
  check_parser_safetensor,
  check_parser_torch,
]

def check_file(filename: str) -> Counter:
  logger.info("checking file: %s", filename)

  counts = Counter()
  for check in ALL_CHECKS:
    logger.info("running check: %s", check.__name__)
    counts.update(check(filename))

  common = counts.most_common()
  logger.info("file %s is most likely: %s", filename, common)

if __name__ == "__main__":
  for file in argv[1:]:
    check_file(file)