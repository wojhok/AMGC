""" Main module for managing program flow."""

import logging
import argparse
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from torchvision import datasets

from model.model import AMGCModel
from trainer.trainer import Trainer
from callbacks.early_stopping import EarlyStopping
from utils.param_by_name import get_optimizer
from utils.set_up_utils import set_device, load_config


def main(config: Dict[str, Any]) -> None:
    """Main function.

    Args:
        config (Dict[str, Any]): Dictionary that contains training configuration.
    """
    device = set_device()
    model = AMGCModel(num_classes=config["num_classes"])
    model.to(device)
    image_shape = config["image_width"], config["image_height"]
    basic_transforms = v2.Compose(
        [v2.Resize(image_shape), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    )
    dataset = datasets.ImageFolder(root=config["dataset_path"], transform=basic_transforms)
    total_size = len(dataset)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, _ = random_split(
        dataset, [train_size, val_size, test_size]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    valid_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    learning_rate = config["learning_rate"]
    optimizer = get_optimizer(config["optimizer"], model.parameters(), learning_rate)
    early_stop_callback = EarlyStopping(config["patience"], config["early_stopping_metric"])
    callbacks = [early_stop_callback]
    trainer = Trainer(
        model,
        train_dataloader,
        valid_dataloader,
        criterion,
        optimizer,
        device,
        config["num_classes"],
        config["epochs"],
        callbacks=callbacks
    )
    trainer.train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("-y", "--yaml", help="Path to config yaml file.")
    args = argument_parser.parse_args()
    config_params = load_config(args.yaml)
    main(config_params)
