import time
from contextlib import ContextDecorator
from typing import Optional


class Timer(ContextDecorator):
    """
    Simple timer utility.

    Example
    -------
    timer = Timer()
    timer.start()
    ...
    timer.stop()

    Example
    -------
    with Timer() as t:
        ...
    print(t.elapsed)

    Example
    -------
    @Timer()
    def func():
        ...
    """

    def __init__(self):
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self.elapsed: float = 0.0

    def start(self) -> None:
        self._start_time = time.perf_counter()
        self._end_time = None

    def stop(self) -> float:
        if self._start_time is None:
            raise RuntimeError(
                "Timer has not been started."
            )

        self._end_time = time.perf_counter()
        self.elapsed = (
            self._end_time - self._start_time
        )

        return self.elapsed

    def reset(self) -> None:
        self._start_time = None
        self._end_time = None
        self.elapsed = 0.0

    @property
    def running(self) -> bool:
        return (
            self._start_time is not None
            and self._end_time is None
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


class AverageTimer:
    """
    Track average execution time.

    Example
    -------
    timer = AverageTimer()

    timer.update(0.5)
    timer.update(0.7)

    print(timer.avg)
    """

    def __init__(self):
        self.reset()

    def update(self, value: float) -> None:
        self.total += value
        self.count += 1

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0

        return self.total / self.count


def time_function(func, *args, **kwargs):
    """
    Measure execution time of a function.

    Returns
    -------
    tuple
        (result, elapsed_time)
    """
    start = time.perf_counter()

    result = func(*args, **kwargs)

    elapsed = (
        time.perf_counter() - start
    )

    return result, elapsed