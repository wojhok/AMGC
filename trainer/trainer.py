"""Module for trainers."""

import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import Precision

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)


class Trainer:
    """Stores functionalities connected with process of training and validating model."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        num_classes: int,
        num_epochs: int,
        callbacks=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_classes = num_classes
        self.writer = SummaryWriter()
        self.callbacks = callbacks if callbacks else []
        self.num_epochs = num_epochs
        self.logs = {}

    def train(self):
        """Manages process of training functions for given number of epochs.

        Args:
            num_epochs (int): Number of epochs.
        """
        self._execute_callbacks("on_train_begin")
        for epoch in range(self.num_epochs):
            self.logs = {"epoch": epoch}
            self._execute_callbacks("on_epoch_begin", epoch, self.logs)
            metrics = self._train_one_epoch(epoch)
            self.logs.update(metrics)
            self.logs['model'] = self.model
            self._execute_callbacks("on_epoch_end", epoch, self.logs)
            if any(
                getattr(callback, "early_stop", True) for callback in self.callbacks
            ):
                logger.info("STOP TRAINING EARLY DUE TO LACK OF IMPROVEMENT")
                break
        self._execute_callbacks("on_train_end")

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss = loss.item()

        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar("Loss/train", avg_loss, epoch)
        logger.info(
            "Epoch: %s/%s, Train Loss: %.4f", epoch + 1, self.num_epochs, avg_loss
        )
        val_metrics = self.validate(epoch)
        metrics = {"val/precision": val_metrics["precision"], "train/loss": avg_loss}
        return metrics

    def validate(self, epoch: int):
        """
        Manages process of validation trained model.

        Args:
            epoch (int): Epoch.
        """
        self.model.eval()
        precision = Precision(
            task="multiclass", num_classes=self.num_classes, average="macro"
        ).to(self.device)
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                predicted = torch.argmax(outputs, dim=1)
                targets = labels
                precision.update(predicted, targets)
        precision_score = precision.compute()
        self.writer.add_scalar("Precision/val", precision_score.item(), epoch)
        logger.info(" Precision: %.2f", precision_score.item())
        metrics = {"precision": precision_score.item()}
        return metrics

    def _execute_callbacks(self, method_name: str, *args, **kwargs):
        for callback in self.callbacks:
            method = getattr(callback, method_name)
            method(*args, *kwargs)
