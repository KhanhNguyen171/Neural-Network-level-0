# src/training/callbacks.py

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


class Callback(ABC):
    """
    Base callback interface.

    All custom callbacks should inherit from this class.
    """

    def on_train_begin(
        self,
        trainer: Any,
    ) -> None:
        pass

    def on_train_end(
        self,
        trainer: Any,
    ) -> None:
        pass

    def on_epoch_begin(
        self,
        trainer: Any,
        epoch: int,
    ) -> None:
        pass

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        logs: Optional[Dict[str, float]] = None,
    ) -> None:
        pass

    def on_batch_begin(
        self,
        trainer: Any,
        batch_idx: int,
    ) -> None:
        pass

    def on_batch_end(
        self,
        trainer: Any,
        batch_idx: int,
        logs: Optional[Dict[str, float]] = None,
    ) -> None:
        pass


class CallbackList:
    """
    Container managing multiple callbacks.

    Example
    -------
    callbacks = CallbackList([
        EarlyStopping(...),
        ModelCheckpoint(...)
    ])
    """

    def __init__(
        self,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        self.callbacks: List[Callback] = callbacks or []

    def append(
        self,
        callback: Callback,
    ) -> None:
        self.callbacks.append(callback)

    def extend(
        self,
        callbacks: List[Callback],
    ) -> None:
        self.callbacks.extend(callbacks)

    def on_train_begin(
        self,
        trainer: Any,
    ) -> None:
        for callback in self.callbacks:
            callback.on_train_begin(trainer)

    def on_train_end(
        self,
        trainer: Any,
    ) -> None:
        for callback in self.callbacks:
            callback.on_train_end(trainer)

    def on_epoch_begin(
        self,
        trainer: Any,
        epoch: int,
    ) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(
                trainer,
                epoch,
            )

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        logs: Optional[Dict[str, float]] = None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(
                trainer,
                epoch,
                logs,
            )

    def on_batch_begin(
        self,
        trainer: Any,
        batch_idx: int,
    ) -> None:
        for callback in self.callbacks:
            callback.on_batch_begin(
                trainer,
                batch_idx,
            )

    def on_batch_end(
        self,
        trainer: Any,
        batch_idx: int,
        logs: Optional[Dict[str, float]] = None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(
                trainer,
                batch_idx,
                logs,
            )

    def __len__(self) -> int:
        return len(self.callbacks)

    def __iter__(self):
        return iter(self.callbacks)

    def __getitem__(
        self,
        index: int,
    ) -> Callback:
        return self.callbacks[index]

    def __repr__(self) -> str:
        names = [
            cb.__class__.__name__
            for cb in self.callbacks
        ]

        return (
            f"CallbackList("
            f"num_callbacks={len(self)}, "
            f"callbacks={names}"
            f")"
        )