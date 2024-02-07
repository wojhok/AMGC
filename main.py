""" Main module for managing program flow."""

import logging
import argparse
from typing import Tuple

import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from torchvision import datasets

from model.model import AMGCModel
from trainer.trainer import Trainer
from callbacks.early_stopping import EarlyStopping


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


def main(images_dir: str, image_shape: Tuple[int, int]) -> None:
    """Main function.

    Args:
        images_dir (str): Path to an directory containing images.
        image_shape (Tuple[int, int]): Shape of image for nn.
    """
    device = set_device()
    model = AMGCModel(num_classes=10)
    model.to(device)
    basic_transforms = v2.Compose(
        [v2.Resize(image_shape), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    )
    dataset = datasets.ImageFolder(root=images_dir, transform=basic_transforms)
    total_size = len(dataset)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, _ = random_split(
        dataset, [train_size, val_size, test_size]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stop_callback = EarlyStopping(10, "val/precision")
    callbacks = [early_stop_callback]
    trainer = Trainer(
        model,
        train_dataloader,
        valid_dataloader,
        criterion,
        optimizer,
        device,
        10,
        100,
        callbacks=callbacks,
    )
    trainer.train(50)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "-id", "--images-dir", help="Path to root_dir of images."
    )
    argument_parser.add_argument(
        "-is", "--images-size", nargs=2, help="Path to root_dir of images.", type=int
    )
    args = argument_parser.parse_args()
    main(args.images_dir, args.images_size)
