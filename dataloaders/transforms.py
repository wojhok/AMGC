"""Module that contains custom transforms implementations"""
from torch import nn


class EmptyTransform(nn.Module):
    """Dummy transform"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Dummy forward no changes applied to input"""
        return x
