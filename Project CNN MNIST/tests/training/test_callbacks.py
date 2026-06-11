# tests/training/test_callbacks.py

from src.training.callbacks import (
    Callback,
    CallbackList,
)

# pytest tests/training/test_callbacks.py -v

# Helpers


class DummyTrainer:
    pass


class RecordingCallback(Callback):
    def __init__(self):
        self.events = []

    def on_train_begin(self, trainer):
        self.events.append("train_begin")

    def on_train_end(self, trainer):
        self.events.append("train_end")

    def on_epoch_begin(self, trainer, epoch):
        self.events.append(
            ("epoch_begin", epoch)
        )

    def on_epoch_end(
        self,
        trainer,
        epoch,
        logs=None,
    ):
        self.events.append(
            ("epoch_end", epoch, logs)
        )

    def on_batch_begin(
        self,
        trainer,
        batch_idx,
    ):
        self.events.append(
            ("batch_begin", batch_idx)
        )

    def on_batch_end(
        self,
        trainer,
        batch_idx,
        logs=None,
    ):
        self.events.append(
            ("batch_end", batch_idx, logs)
        )


# Callback Base Class


def test_callback_can_be_instantiated():
    class MyCallback(Callback):
        pass

    callback = MyCallback()

    assert isinstance(callback, Callback)


def test_default_callback_methods_do_not_fail():
    class MyCallback(Callback):
        pass

    callback = MyCallback()
    trainer = DummyTrainer()

    callback.on_train_begin(trainer)
    callback.on_train_end(trainer)

    callback.on_epoch_begin(
        trainer,
        epoch=1,
    )

    callback.on_epoch_end(
        trainer,
        epoch=1,
        logs={},
    )

    callback.on_batch_begin(
        trainer,
        batch_idx=0,
    )

    callback.on_batch_end(
        trainer,
        batch_idx=0,
        logs={},
    )


# CallbackList Initialization


def test_callback_list_empty():
    callbacks = CallbackList()

    assert len(callbacks) == 0


def test_callback_list_with_callbacks():
    callbacks = CallbackList(
        [
            RecordingCallback(),
            RecordingCallback(),
        ]
    )

    assert len(callbacks) == 2


# Append / Extend


def test_append_callback():
    callbacks = CallbackList()

    callbacks.append(
        RecordingCallback()
    )

    assert len(callbacks) == 1


def test_extend_callbacks():
    callbacks = CallbackList()

    callbacks.extend(
        [
            RecordingCallback(),
            RecordingCallback(),
        ]
    )

    assert len(callbacks) == 2


# Event Dispatch


def test_on_train_begin_dispatch():
    cb = RecordingCallback()

    callbacks = CallbackList([cb])

    callbacks.on_train_begin(
        DummyTrainer()
    )

    assert cb.events == [
        "train_begin"
    ]


def test_on_train_end_dispatch():
    cb = RecordingCallback()

    callbacks = CallbackList([cb])

    callbacks.on_train_end(
        DummyTrainer()
    )

    assert cb.events == [
        "train_end"
    ]


def test_on_epoch_begin_dispatch():
    cb = RecordingCallback()

    callbacks = CallbackList([cb])

    callbacks.on_epoch_begin(
        DummyTrainer(),
        epoch=5,
    )

    assert cb.events == [
        ("epoch_begin", 5)
    ]


def test_on_epoch_end_dispatch():
    cb = RecordingCallback()

    callbacks = CallbackList([cb])

    logs = {
        "loss": 0.25
    }

    callbacks.on_epoch_end(
        DummyTrainer(),
        epoch=3,
        logs=logs,
    )

    assert cb.events == [
        ("epoch_end", 3, logs)
    ]


def test_on_batch_begin_dispatch():
    cb = RecordingCallback()

    callbacks = CallbackList([cb])

    callbacks.on_batch_begin(
        DummyTrainer(),
        batch_idx=7,
    )

    assert cb.events == [
        ("batch_begin", 7)
    ]


def test_on_batch_end_dispatch():
    cb = RecordingCallback()

    callbacks = CallbackList([cb])

    logs = {
        "loss": 0.1
    }

    callbacks.on_batch_end(
        DummyTrainer(),
        batch_idx=7,
        logs=logs,
    )

    assert cb.events == [
        ("batch_end", 7, logs)
    ]


# Multiple Callbacks


def test_multiple_callbacks_receive_event():
    cb1 = RecordingCallback()
    cb2 = RecordingCallback()

    callbacks = CallbackList(
        [cb1, cb2]
    )

    callbacks.on_train_begin(
        DummyTrainer()
    )

    assert cb1.events == [
        "train_begin"
    ]

    assert cb2.events == [
        "train_begin"
    ]


def test_multiple_callbacks_epoch_end():
    cb1 = RecordingCallback()
    cb2 = RecordingCallback()

    callbacks = CallbackList(
        [cb1, cb2]
    )

    logs = {
        "accuracy": 0.95
    }

    callbacks.on_epoch_end(
        DummyTrainer(),
        epoch=2,
        logs=logs,
    )

    assert cb1.events == [
        ("epoch_end", 2, logs)
    ]

    assert cb2.events == [
        ("epoch_end", 2, logs)
    ]


# Iteration


def test_callback_list_iterable():
    cb1 = RecordingCallback()
    cb2 = RecordingCallback()

    callbacks = CallbackList(
        [cb1, cb2]
    )

    collected = list(callbacks)

    assert collected == [
        cb1,
        cb2,
    ]


# Indexing


def test_callback_list_getitem():
    cb1 = RecordingCallback()
    cb2 = RecordingCallback()

    callbacks = CallbackList(
        [cb1, cb2]
    )

    assert callbacks[0] is cb1
    assert callbacks[1] is cb2


# Repr


def test_callback_list_repr_empty():
    callbacks = CallbackList()

    rep = repr(callbacks)

    assert "CallbackList" in rep
    assert "num_callbacks=0" in rep


def test_callback_list_repr_non_empty():
    callbacks = CallbackList(
        [
            RecordingCallback(),
            RecordingCallback(),
        ]
    )

    rep = repr(callbacks)

    assert "CallbackList" in rep
    assert "num_callbacks=2" in rep
    assert "RecordingCallback" in rep