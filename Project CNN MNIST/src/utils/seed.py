import random
from typing import Optional

import numpy as np
import torch


def set_seed(
    seed: int = 42,
    deterministic: bool = True,
) -> int:
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value.
    deterministic : bool
        Enable deterministic algorithms.

    Returns
    -------
    int
        Seed value used.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    return seed


def seed_worker(worker_id: int) -> None:
    """
    Seed DataLoader worker.

    Example
    -------
    DataLoader(
        dataset,
        worker_init_fn=seed_worker
    )
    """
    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_generator(
    seed: int = 42,
) -> torch.Generator:
    """
    Create reproducible torch Generator.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def get_seed() -> Optional[int]:
    """
    Return current torch initial seed.

    Returns
    -------
    int
    """
    return torch.initial_seed()


def enable_deterministic() -> None:
    """
    Enable deterministic behavior.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def disable_deterministic() -> None:
    """
    Disable deterministic behavior.
    """
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass