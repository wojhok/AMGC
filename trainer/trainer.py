"""Module for trainers."""

import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import Precision, Recall, Accuracy, F1Score
import hyperopt
from tqdm import tqdm

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
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        callbacks=None,
        tune=False
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
        self.lr_scheduler = lr_scheduler
        self.logs = {}
        self.val_precision = Precision(
            num_classes=num_classes, average="macro", task="multiclass"
        ).to(device)
        self.val_recall = Recall(
            num_classes=num_classes, average="macro", task="multiclass"
        ).to(device)
        self.val_accuracy = Accuracy(
            num_classes=num_classes, average="macro", task="multiclass"
        ).to(device)
        self.val_f1_score = F1Score(
            num_classes=num_classes, average="macro", task="multiclass"
        ).to(device)
        self.tune = tune

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

            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(self.logs["val/loss"])
            else:
                self.lr_scheduler.step()
            self._execute_callbacks("on_epoch_end", epoch, self.logs)

            if any(
                getattr(callback, "early_stop", True) for callback in self.callbacks
            ):
                logger.info("STOP TRAINING EARLY DUE TO LACK OF IMPROVEMENT")
                if self.tune:
                    return {'loss': self.logs["val/loss"], 'status': hyperopt.STATUS_OK}
                break
        self._execute_callbacks("on_train_end")
        if self.tune:
            return {'loss': self.logs["val/loss"], 'status': hyperopt.STATUS_OK}

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch: {epoch+1}/{self.num_epochs}", unit='batch'
        )
        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar("Loss/train", avg_loss, epoch)
        logger.info(
            "Epoch: %s/%s, Train Loss: %.4f", epoch + 1, self.num_epochs, avg_loss
        )
        val_metrics = self.validate(epoch)
        metrics = {
            "val/precision": val_metrics["precision"],
            "val/recall": val_metrics["recall"],
            "val/f1_score": val_metrics["f1_score"],
            "val/accuracy": val_metrics["accuracy"],
            "val/loss": val_metrics["loss"],
            "train/loss": avg_loss
        }
        return metrics

    def validate(self, epoch: int):
        """
        Manages process of validation trained model.

        Args:
            epoch (int): Epoch.
        """
        self.model.eval()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_accuracy.reset()
        self.val_f1_score.reset()
        total_loss = 0.0
        progress_bar = tqdm(self.val_loader, desc='Validating', unit="batch")
        with torch.no_grad():
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                predicted = torch.argmax(outputs, dim=1)
                self.val_precision.update(predicted, labels)
                self.val_f1_score.update(predicted, labels)
                self.val_recall.update(predicted, labels)
                self.val_accuracy.update(predicted, labels)
        avg_loss = total_loss / len(self.val_loader)
        precision_score = self.val_precision.compute().item()
        recall_score = self.val_recall.compute().item()
        f1_score = self.val_f1_score.compute().item()
        accuracy_score = self.val_accuracy.compute().item()
        self.writer.add_scalar("Precision/val", precision_score, epoch)
        self.writer.add_scalar("Recall/val", recall_score, epoch)
        self.writer.add_scalar("F1_score/val", f1_score, epoch)
        self.writer.add_scalar("Accuracy/val", accuracy_score, epoch)
        self.writer.add_scalar("Loss/val", avg_loss, epoch)
        logger.info(
            "Validation - Loss: %.4f, Precision: %.2f, Recall: %.2f, F1 Score: %.2f, "
            "Accuracy: %.2f",
            avg_loss,
            precision_score,
            recall_score,
            f1_score,
            accuracy_score
        )
        metrics = {
            "loss": avg_loss,
            "precision": precision_score,
            "recall": recall_score,
            "f1_score": f1_score,
            "accuracy": accuracy_score
        }
        return metrics

    def _execute_callbacks(self, method_name: str, *args, **kwargs):
        for callback in self.callbacks:
            method = getattr(callback, method_name)
            if method:
                method(*args, **kwargs)
