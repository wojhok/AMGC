"""Module for trainers."""
import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.classification import Precision


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# Set the default log level to INFO
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
        num_classes: int
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_classes = num_classes

    def train(self, num_epochs: int):
        """Manages process of training functions for given number of epochs.

        Args:
            num_epochs (int): Number of epochs.
        """
        for epoch in range(num_epochs):
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

            print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}")

    def validate(self):
        """Manages process of validation trained model."""
        self.model.eval()
        precision = Precision(task="multiclass", num_classes=self.num_classes, average="macro")
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                predicted = torch.argmax(outputs, dim=1)
                targets = torch.argmax(outputs, dim=1)
                precision.update(predicted, targets)
        precision_score = precision.compute()
        logger.info("Precision: %.2f", precision_score)
