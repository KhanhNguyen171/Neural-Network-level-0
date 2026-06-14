from typing import Any, Callable, Dict, List


class Registry:
    """
    Generic registry for models, datasets, losses, etc.
    """

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Any] = {}

    def register(
        self,
        name: str = None,
    ) -> Callable:
        """
        Decorator-based registration.

        Example
        -------
        @registry.register()
        class MyModel:
            pass

        @registry.register("custom_name")
        class MyModel:
            pass
        """

        def decorator(obj):
            key = name or obj.__name__.lower()

            if key in self._registry:
                raise KeyError(
                    f"'{key}' already registered in '{self.name}'"
                )

            self._registry[key] = obj
            return obj

        return decorator

    def add(
        self,
        name: str,
        obj: Any,
    ) -> None:
        """
        Register object manually.
        """
        if name in self._registry:
            raise KeyError(
                f"'{name}' already registered in '{self.name}'"
            )

        self._registry[name] = obj

    def get(self, name: str) -> Any:
        """
        Retrieve registered object.
        """
        if name not in self._registry:
            raise KeyError(
                f"'{name}' not found in '{self.name}' registry"
            )

        return self._registry[name]

    def remove(self, name: str) -> None:
        """
        Remove object from registry.
        """
        if name not in self._registry:
            raise KeyError(
                f"'{name}' not found in '{self.name}' registry"
            )

        del self._registry[name]

    def contains(self, name: str) -> bool:
        """
        Check whether key exists.
        """
        return name in self._registry

    def keys(self) -> List[str]:
        """
        Return all registered names.
        """
        return list(self._registry.keys())

    def values(self) -> List[Any]:
        """
        Return all registered objects.
        """
        return list(self._registry.values())

    def items(self):
        """
        Return registry items.
        """
        return self._registry.items()

    def clear(self) -> None:
        """
        Remove all entries.
        """
        self._registry.clear()

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, item: str) -> bool:
        return item in self._registry

    def __getitem__(self, item: str):
        return self.get(item)

    def __repr__(self) -> str:
        return (
            f"Registry("
            f"name='{self.name}', "
            f"size={len(self)}"
            f")"
        )