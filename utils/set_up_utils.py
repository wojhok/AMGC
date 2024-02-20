"""
This module contains utility functions to assist in setting up the environment and configurations
for PyTorch models.
"""
import logging

import torch
import yaml

def set_device():
    """Checks if torch backend is available."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(" Device: %s", device)
    return device

def load_config(config_path: str):
    """Loads config for training.

    Args:
        config_path (str): Path to config file.

    Returns:
        Dict: Configuration for training.
    """
    with open(config_path, 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config
