"""Module for creating objects of parameters by class."""
from typing import Iterable, Dict, Any
import inspect

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
    ReduceLROnPlateau, OneCycleLR, CosineAnnealingWarmRestarts
)


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


def get_learning_rate_scheduler(config: Dict[str, Any], optimizer: optim.Optimizer):
    """Get learning rate scheduler by name and initialize it with the optimizer.

    Args:
        scheduler_name (Dict[str, Any]): Dictionary containing lr-scheduler and its configuration.
        optimizer (optim.Optimizer): Optimizer linked with the scheduler.

    Returns:
        Any: Learning rate scheduler object.
    """
    scheduler_config = {
        key.replace('scheduler_', ''): value for key, value in config.items() if 'scheduler' in key
    }
    scheduler_type = scheduler_config.get('type')

    if not scheduler_type:
        raise ValueError(
            "Scheduler type must be specified in the configuration.")

    scheduler_classes = {
        "StepLR": StepLR,
        "MultiStepLR": MultiStepLR,
        "ExponentialLR": ExponentialLR,
        "CosineAnnealingLR": CosineAnnealingLR,
        "ReduceLROnPlateau": ReduceLROnPlateau,
        "OneCycleLR": OneCycleLR,
        "CosineAnnealingWarmRestarts": CosineAnnealingWarmRestarts
    }
    scheduler_class = scheduler_classes.get(scheduler_type)
    if not scheduler_class:
        raise ValueError(f"Scheduler '{scheduler_type}' is not recognized.")

    scheduler_params = {
        k: v
        for k, v in scheduler_config.items()
        if k != "type" and k in inspect.getfullargspec(scheduler_class.__init__).args
    }

    return scheduler_class(optimizer, **scheduler_params)
