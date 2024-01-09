from typing import Optional


class RetryException(Exception):
    """
    Used when a chain pipeline has run out of retries.
    """

    pass


class CancelledException(Exception):
    """
    Used when a job has been cancelled and needs to stop.
    """

    reason: Optional[str]

    def __init__(self, *args: object, reason: Optional[str] = None) -> None:
        super().__init__(*args)

        self.reason = reason


class RequestException(Exception):
    """
    Used when an HTTP request has failed.
    """

    pass
