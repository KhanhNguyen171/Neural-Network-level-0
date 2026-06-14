"""
Device utilities.
"""

from typing import Any

import torch


def get_device() -> torch.device:
    """
    Automatically select device.

    Returns
    -------
    torch.device
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def gpu_available() -> bool:
    """
    Check CUDA availability.
    """
    return torch.cuda.is_available()


def device_name() -> str:
    """
    Get active device name.
    """

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)

    return "CPU"


def move_to_device(
    obj: Any,
    device: torch.device | str,
):
    """
    Move tensor / module / collection
    recursively to device.
    """

    if isinstance(obj, torch.Tensor):
        return obj.to(device)

    if isinstance(obj, torch.nn.Module):
        return obj.to(device)

    if isinstance(obj, dict):
        return {
            k: move_to_device(v, device)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return type(obj)(
            move_to_device(x, device)
            for x in obj
        )

    return obj


def model_device(
    model: torch.nn.Module,
) -> torch.device:
    """
    Get model device.
    """

    return next(model.parameters()).device


def synchronize() -> None:
    """
    Synchronize CUDA operations.
    Useful for benchmarking.
    """

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def memory_allocated_mb() -> float:
    """
    Current allocated GPU memory (MB).
    """

    if not torch.cuda.is_available():
        return 0.0

    return (
        torch.cuda.memory_allocated()
        / 1024**2
    )


def memory_reserved_mb() -> float:
    """
    Current reserved GPU memory (MB).
    """

    if not torch.cuda.is_available():
        return 0.0

    return (
        torch.cuda.memory_reserved()
        / 1024**2
    )