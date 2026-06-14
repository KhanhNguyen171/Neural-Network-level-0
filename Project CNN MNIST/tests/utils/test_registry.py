import pytest

from src.utils.registry import Registry

# pytest tests/utils/test_registry.py -v


def test_create_registry():
    registry = Registry("models")

    assert registry.name == "models"
    assert len(registry) == 0


def test_manual_add_and_get():
    registry = Registry("models")

    class Dummy:
        pass

    registry.add("dummy", Dummy)

    assert registry.get("dummy") is Dummy


def test_add_duplicate_raises():
    registry = Registry("models")

    class Dummy:
        pass

    registry.add("dummy", Dummy)

    with pytest.raises(KeyError):
        registry.add("dummy", Dummy)


def test_get_missing_key():
    registry = Registry("models")

    with pytest.raises(KeyError):
        registry.get("missing")


def test_remove():
    registry = Registry("models")

    class Dummy:
        pass

    registry.add("dummy", Dummy)

    registry.remove("dummy")

    assert len(registry) == 0


def test_remove_missing():
    registry = Registry("models")

    with pytest.raises(KeyError):
        registry.remove("missing")


def test_contains():
    registry = Registry("models")

    class Dummy:
        pass

    registry.add("dummy", Dummy)

    assert registry.contains("dummy")
    assert "dummy" in registry


def test_keys():
    registry = Registry("models")

    registry.add("a", object)
    registry.add("b", list)

    keys = registry.keys()

    assert "a" in keys
    assert "b" in keys


def test_values():
    registry = Registry("models")

    registry.add("a", object)
    registry.add("b", list)

    values = registry.values()

    assert object in values
    assert list in values


def test_items():
    registry = Registry("models")

    registry.add("a", object)

    items = dict(registry.items())

    assert items["a"] is object


def test_clear():
    registry = Registry("models")

    registry.add("a", object)
    registry.add("b", list)

    registry.clear()

    assert len(registry) == 0


def test_getitem():
    registry = Registry("models")

    registry.add("dummy", object)

    assert registry["dummy"] is object


def test_register_decorator_default_name():
    registry = Registry("models")

    @registry.register()
    class MyModel:
        pass

    assert registry.get("mymodel") is MyModel


def test_register_decorator_custom_name():
    registry = Registry("models")

    @registry.register("cnn")
    class MyModel:
        pass

    assert registry.get("cnn") is MyModel


def test_register_duplicate_decorator():
    registry = Registry("models")

    @registry.register("cnn")
    class ModelA:
        pass

    with pytest.raises(KeyError):

        @registry.register("cnn")
        class ModelB:
            pass


def test_repr():
    registry = Registry("models")

    text = repr(registry)

    assert "Registry" in text
    assert "models" in text