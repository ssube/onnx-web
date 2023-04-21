from threading import Timer


class Interval(Timer):
    """
    From https://stackoverflow.com/a/48741004
    """

    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)
