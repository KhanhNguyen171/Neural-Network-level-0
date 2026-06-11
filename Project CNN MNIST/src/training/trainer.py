# src/training/trainer.py

from __future__ import annotations

from typing import Dict
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from .callbacks import CallbackList
from .history import TrainingHistory
from .metrics import AverageMeter
from .metrics import accuracy_score


class Trainer:
    """
    Generic PyTorch trainer.

    Supports:
    - training
    - validation
    - history tracking
    - callbacks
    - checkpointing
    - early stopping

    Example
    -------
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device="cuda",
    )

    trainer.fit(
        train_loader,
        val_loader,
        epochs=10,
    )
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        scheduler=None,
        callbacks: Optional[CallbackList] = None,
        checkpoint=None,
        early_stopping=None,
    ) -> None:

        self.model = model

        self.criterion = criterion

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.device = torch.device(device)

        self.callbacks = (
            callbacks
            if callbacks is not None
            else CallbackList()
        )

        self.checkpoint = checkpoint

        self.early_stopping = early_stopping

        self.history = TrainingHistory()

        self.current_epoch = 0

        self.model.to(self.device)


    # Training


    def train_one_epoch(
        self,
        train_loader: DataLoader,
    ) -> Dict[str, float]:

        self.model.train()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        for batch_idx, (inputs, targets) in enumerate(
            train_loader
        ):

            self.callbacks.on_batch_begin(
                self,
                batch_idx,
            )

            inputs = inputs.to(self.device)

            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.criterion(
                outputs,
                targets,
            )

            loss.backward()

            self.optimizer.step()

            batch_size = inputs.size(0)

            acc = accuracy_score(
                outputs.detach(),
                targets,
            )

            loss_meter.update(
                loss.item(),
                batch_size,
            )

            acc_meter.update(
                acc,
                batch_size,
            )

            self.callbacks.on_batch_end(
                self,
                batch_idx,
                {
                    "loss": loss.item(),
                    "accuracy": acc,
                },
            )

        return {
            "loss": loss_meter.avg,
            "accuracy": acc_meter.avg,
        }


    # Validation


    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:

        self.model.eval()

        loss_meter = AverageMeter()

        acc_meter = AverageMeter()

        for inputs, targets in val_loader:

            inputs = inputs.to(self.device)

            targets = targets.to(self.device)

            outputs = self.model(inputs)

            loss = self.criterion(
                outputs,
                targets,
            )

            batch_size = inputs.size(0)

            acc = accuracy_score(
                outputs,
                targets,
            )

            loss_meter.update(
                loss.item(),
                batch_size,
            )

            acc_meter.update(
                acc,
                batch_size,
            )

        return {
            "loss": loss_meter.avg,
            "accuracy": acc_meter.avg,
        }


    # Fit


    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[
            DataLoader
        ] = None,
        epochs: int = 1,
    ) -> TrainingHistory:

        self.callbacks.on_train_begin(
            self
        )

        for epoch in range(epochs):

            self.current_epoch = epoch + 1

            self.callbacks.on_epoch_begin(
                self,
                self.current_epoch,
            )

            train_metrics = (
                self.train_one_epoch(
                    train_loader
                )
            )

            logs = {
                f"train_{k}": v
                for k, v in train_metrics.items()
            }

            # Validation
            if val_loader is not None:

                val_metrics = self.validate(
                    val_loader
                )

                logs.update(
                    {
                        f"val_{k}": v
                        for k, v in val_metrics.items()
                    }
                )

            # Scheduler step
            if self.scheduler is not None:

                try:
                    self.scheduler.step(
                        logs.get(
                            "val_loss"
                        )
                    )
                except TypeError:
                    self.scheduler.step()

            # History update
            self.history.update(
                **logs
            )

            # Callback
            self.callbacks.on_epoch_end(
                self,
                self.current_epoch,
                logs,
            )

            # Checkpoint
            if self.checkpoint is not None:

                self.checkpoint.save_last(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    history=self.history,
                    epoch=self.current_epoch,
                )

            # Early stopping
            if (
                self.early_stopping
                is not None
                and "val_loss" in logs
            ):

                stop = (
                    self.early_stopping.step(
                        current=logs[
                            "val_loss"
                        ],
                        epoch=self.current_epoch,
                    )
                )

                if stop:
                    break

        self.callbacks.on_train_end(
            self
        )

        return self.history


    # Prediction


    @torch.no_grad()
    def predict(
        self,
        dataloader: DataLoader,
    ) -> torch.Tensor:

        self.model.eval()

        predictions = []

        for inputs, _ in dataloader:

            inputs = inputs.to(
                self.device
            )

            outputs = self.model(
                inputs
            )

            preds = outputs.argmax(
                dim=1
            )

            predictions.append(
                preds.cpu()
            )

        return torch.cat(
            predictions,
            dim=0,
        )


    # Evaluation


    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:

        return self.validate(
            dataloader
        )


    # Checkpoint Resume


    def resume(
        self,
        checkpoint_path: str,
    ) -> Dict:

        if self.checkpoint is None:
            raise RuntimeError(
                "Checkpoint manager "
                "is not configured."
            )

        checkpoint = (
            self.checkpoint.load(
                filepath=checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                history=self.history,
            )
        )

        self.current_epoch = (
            checkpoint.get(
                "epoch",
                0,
            )
        )

        return checkpoint


    # Utilities


    @property
    def epoch(self) -> int:
        return self.current_epoch

    def __repr__(self) -> str:

        return (
            f"Trainer("
            f"model={self.model.__class__.__name__}, "
            f"device='{self.device}', "
            f"epoch={self.current_epoch}"
            f")"
        )