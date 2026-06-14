import random

import numpy as np
import pytest
import torch

from src.utils.seed import (
    set_seed,
    seed_worker,
    create_generator,
    get_seed,
    enable_deterministic,
    disable_deterministic,
)

# pytest tests/utils/test_seed.py -v

def test_set_seed_returns_seed():
    seed = set_seed(123)

    assert seed == 123


def test_set_seed_reproducibility_random():
    set_seed(123)
    value1 = random.random()

    set_seed(123)
    value2 = random.random()

    assert value1 == value2


def test_set_seed_reproducibility_numpy():
    set_seed(123)
    value1 = np.random.rand()

    set_seed(123)
    value2 = np.random.rand()

    assert value1 == value2


def test_set_seed_reproducibility_torch():
    set_seed(123)
    value1 = torch.rand(5)

    set_seed(123)
    value2 = torch.rand(5)

    assert torch.equal(value1, value2)


def test_create_generator():
    generator = create_generator(123)

    assert isinstance(generator, torch.Generator)


def test_create_generator_reproducibility():
    gen1 = create_generator(123)
    gen2 = create_generator(123)

    x1 = torch.rand(5, generator=gen1)
    x2 = torch.rand(5, generator=gen2)

    assert torch.equal(x1, x2)


def test_get_seed():
    seed = get_seed()

    assert isinstance(seed, int)
    assert seed > 0


def test_seed_worker_runs():
    seed_worker(0)


def test_enable_deterministic():
    enable_deterministic()

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_disable_deterministic():
    disable_deterministic()

    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True


def test_multiple_seed_calls():
    set_seed(1)
    a = torch.rand(3)

    set_seed(2)
    b = torch.rand(3)

    assert not torch.equal(a, b)


@pytest.mark.parametrize(
    "seed",
    [0, 1, 42, 123, 999],
)
def test_various_seeds(seed):
    result = set_seed(seed)

    assert result == seed