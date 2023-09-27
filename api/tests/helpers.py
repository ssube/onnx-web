from typing import List


def test_with_models(models: List[str]):
  def wrapper(func):
    # TODO: check if models exist
    return func

  return wrapper
