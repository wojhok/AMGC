"""
This module defines the AMGCModel, a neural network architecture that combines a ResNet-based
feature extractor with a GRU (Gated Recurrent Unit) layer for advanced image and sequential data
processing. The model is designed to handle complex tasks that require understanding both spatial
features (via ResNet) and temporal dynamics (via GRU)."""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class AMGCModel(nn.Module):
    """
    A neural network model that integrates ResNet and GRU for complex feature extraction.
    This model is suitable for tasks that require understanding of both image features and
    sequential patterns.

    The ResNet component is used for initial image feature extraction, followed by a GRU layer
    that processes sequential data. The outputs of both components are combined and passed through
    fully connected layers for final classification.

    Attributes:
        resnet (nn.Module): The ResNet model for image feature extraction.
        gru (nn.GRU): The GRU layer for processing sequential data.
        gru_fc (nn.Linear): A fully connected layer that follows the GRU layer.
        combine (nn.Sequential): Sequential layers to combine and classify the features.

    Args:
        num_classes (int): The number of classes for the final output layer.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.BatchNorm2d(256),
        )
        self.gru = nn.GRU(
            input_size=224, hidden_size=256, num_layers=1, bidirectional=True
        )
        self.gru_fc = nn.Linear(in_features=512, out_features=256)
        self.combine = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, num_classes))

    def forward(self, x: torch.Tensor):
        """
        Defines the forward pass of the AMGCModel.

        The input is first passed through the ResNet component for feature extraction.
        The output is then processed and passed through the GRU layer.
        Finally, the combined features from ResNet and GRU are used for classification.

        Args:
            x (torch.Tensor): The input tensor. Shape should match the expected input shape of
                ResNet.

        Returns:
            torch.Tensor: The output tensor after processing through the model. Shape corresponds to
            (batch_size, num_classes).
        """
        resnet_out = self.resnet(x)

        x_gru = x.view(x.size(0), -1, 224)
        gru_out, _ = self.gru(x_gru)
        gru_out = self.gru_fc(gru_out[:, -1, :])

        combined = torch.cat((resnet_out, gru_out), dim=1)

        out = self.combine(combined)
        return out


class Reshape(nn.Module):
    """
    A custom PyTorch module for reshaping input tensors.

    This module provides a convenient way to reshape tensors in a neural network pipeline.
    It can be particularly useful when tensors need to be reshaped between different layers
    in a model, allowing for seamless integration into `nn.Sequential` pipelines.

    Attributes:
        shape (tuple): The desired shape to which the input tensors will be reshaped.

    Args:
        *shape (int): Variable length argument list representing the target shape.
    """

    def __init__(self, *shape) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor):
        """
        Reshapes the input tensor to the specified shape.

        Args:
            x (torch.Tensor): The input tensor to be reshaped.

        Returns:
            torch.Tensor: The reshaped tensor.
        """
        return x.view(*self.shape)
