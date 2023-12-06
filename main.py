""" Main module for managing program flow."""
import logging

import torch

from model.model import AMGCModel

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

def main():
    """Main function."""
    device = set_device()
    model = AMGCModel(num_classes=10)
    logging.info(" Model architecture: %s", model)
    model.to(device)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
