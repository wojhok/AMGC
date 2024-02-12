"""Module for creating objects of parameters by class."""
from typing import Iterable

import torch
import torch.optim as optim


def get_optimizer(
    optimizer_name: str,
    model_params: Iterable[torch.nn.parameter.Parameter],
    lr: float,  **kwargs
):
    """Get optimizer class object by name and fill it with parameters passed in lr arg and **kwargs.

    Args:
        optimizer_name (str): Name of the optimizer.
        model_params (Iterable[torch.nn.parameter.Parameter]): Parameters of a model.
        lr (float): Value of learning rate.

    Returns:
        optim.Optimizer: Optimizer object.
    """
    optimizer_names = {
        "adadelta": "Adadelta",
        "adagrad": "Adagrad",
        "adam": "Adam",
        "adamw": "AdamW",
        "sparseadam": "SparseAdam",
        "adamax": "Adamax",
        "asgd": "ASGD",
        "sgd": "SGD",
        "radam": "RAdam",
        "rprop": "Rprop",
        "rmsprop": "RMSprop",
        "nadam": "NAdam",
        "lbfgs": "LBFGS"
    }
    optimizer_class = getattr(
        optim, optimizer_names[optimizer_name.lower()], None)
    if optimizer_class is None:
        raise ValueError(f"Optimizer '{optimizer_name}' is not recognized.")
    return optimizer_class(model_params, lr=lr, **kwargs)
