class RetryException(Exception):
    """
    Used when a chain pipeline has run out of retries.
    """

    pass


class CancelledException(Exception):
    """
    Used when a job has been cancelled and needs to stop.
    """

    pass


class RequestException(Exception):
    """
    Used when an HTTP request has failed.
    """

    pass
