import torch

from src.utils.device import (
    get_device,
    gpu_available,
    device_name,
    move_to_device,
    model_device,
    memory_allocated_mb,
    memory_reserved_mb,
)

# pytest tests/utils/test_device.py -v

def test_get_device():
    device = get_device()

    assert isinstance(
        device,
        torch.device,
    )

    assert device.type in [
        "cpu",
        "cuda",
    ]


def test_gpu_available():
    result = gpu_available()

    assert isinstance(
        result,
        bool,
    )


def test_device_name():
    name = device_name()

    assert isinstance(
        name,
        str,
    )

    assert len(name) > 0


def test_move_tensor_to_device():
    tensor = torch.randn(
        4,
        5,
    )

    device = get_device()

    moved = move_to_device(
        tensor,
        device,
    )

    assert moved.device.type == device.type


def test_move_dict_to_device():
    batch = {
        "x": torch.randn(2, 3),
        "y": torch.tensor([0, 1]),
    }

    device = get_device()

    moved = move_to_device(
        batch,
        device,
    )

    assert moved["x"].device.type == device.type
    assert moved["y"].device.type == device.type


def test_move_list_to_device():
    values = [
        torch.randn(2),
        torch.randn(3),
    ]

    device = get_device()

    moved = move_to_device(
        values,
        device,
    )

    for tensor in moved:
        assert tensor.device.type == device.type


def test_model_device():
    model = torch.nn.Linear(
        10,
        2,
    )

    device = get_device()

    model = model.to(device)

    assert model_device(
        model
    ).type == device.type


def test_memory_allocated_mb():
    value = memory_allocated_mb()

    assert isinstance(
        value,
        float,
    )

    assert value >= 0.0


def test_memory_reserved_mb():
    value = memory_reserved_mb()

    assert isinstance(
        value,
        float,
    )

    assert value >= 0.0