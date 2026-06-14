import time

import pytest

from src.utils.timer import (
    Timer,
    AverageTimer,
    time_function,
)

# pytest tests/utils/test_timer.py -v

def test_timer_start_stop():
    timer = Timer()

    timer.start()

    time.sleep(0.01)

    elapsed = timer.stop()

    assert elapsed > 0
    assert timer.elapsed > 0


def test_timer_stop_without_start():
    timer = Timer()

    with pytest.raises(RuntimeError):
        timer.stop()


def test_timer_reset():
    timer = Timer()

    timer.start()
    time.sleep(0.01)
    timer.stop()

    timer.reset()

    assert timer.elapsed == 0.0
    assert timer._start_time is None
    assert timer._end_time is None


def test_timer_running_property():
    timer = Timer()

    assert timer.running is False

    timer.start()

    assert timer.running is True

    timer.stop()

    assert timer.running is False


def test_timer_context_manager():
    with Timer() as timer:
        time.sleep(0.01)

    assert timer.elapsed > 0


def test_average_timer_initial():
    timer = AverageTimer()

    assert timer.avg == 0.0
    assert timer.count == 0


def test_average_timer_update():
    timer = AverageTimer()

    timer.update(1.0)
    timer.update(3.0)

    assert timer.count == 2
    assert timer.avg == 2.0


def test_average_timer_reset():
    timer = AverageTimer()

    timer.update(1.0)
    timer.update(2.0)

    timer.reset()

    assert timer.count == 0
    assert timer.total == 0.0
    assert timer.avg == 0.0


def test_time_function():
    def square(x):
        time.sleep(0.01)
        return x * x

    result, elapsed = time_function(
        square,
        5,
    )

    assert result == 25
    assert elapsed > 0


def test_time_function_multiple_calls():
    def add(a, b):
        return a + b

    result, elapsed = time_function(
        add,
        2,
        3,
    )

    assert result == 5
    assert elapsed >= 0


def test_average_timer_many_updates():
    timer = AverageTimer()

    for i in range(1, 6):
        timer.update(i)

    assert timer.count == 5
    assert timer.avg == 3.0