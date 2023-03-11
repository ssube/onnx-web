from logging import getLogger
from typing import Any, List, Tuple

logger = getLogger(__name__)


class ModelCache:
    cache: List[Tuple[str, Any, Any]]
    limit: int

    def __init__(self, limit: int) -> None:
        self.cache = []
        self.limit = limit
        logger.debug("creating model cache with limit of %s models", limit)

    def drop(self, tag: str, key: Any) -> int:
        logger.debug("dropping item from cache: %s %s", tag, key)
        removed = [model for model in self.cache if model[0] == tag and model[1] == key]
        for item in removed:
            self.cache.remove(item)

        return len(removed)

    def get(self, tag: str, key: Any) -> Any:
        for t, k, v in self.cache:
            if tag == t and key == k:
                logger.debug("found cached model: %s %s", tag, key)
                return v

        logger.debug("model not found in cache: %s %s", tag, key)
        return None

    def set(self, tag: str, key: Any, value: Any) -> None:
        if self.limit == 0:
            logger.debug("cache limit set to 0, not caching model: %s", tag)
            return

        for i in range(len(self.cache)):
            t, k, v = self.cache[i]
            if tag == t and key != k:
                logger.debug("updating model cache: %s %s", tag, key)
                self.cache[i] = (tag, key, value)
                return

        logger.debug("adding new model to cache: %s %s", tag, key)
        self.cache.append((tag, key, value))
        self.prune()

    def prune(self):
        total = len(self.cache)
        overage = total - self.limit
        if overage > 0:
            removed = self.cache[:overage]
            logger.info(
                "removing %s of %s models from cache, %s",
                overage,
                total,
                [m[0] for m in removed],
            )
            self.cache[:] = self.cache[-self.limit :]
        else:
            logger.debug("model cache below limit, %s of %s", total, self.limit)

    @property
    def size(self):
        return len(self.cache)
