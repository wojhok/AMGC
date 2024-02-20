"""Module that handles hyperparameter optimization."""
from typing import Dict, Any, List
import argparse

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from hyperopt import hp, fmin, tpe, Trials

from model.model import AMGCModel
from trainer.trainer import Trainer
from callbacks.early_stopping import EarlyStopping
from utils.param_by_name import get_optimizer
from utils.set_up_utils import set_device, load_config


def objective(hyperparams: Dict[str, Any]) -> None:
    """
    Objective function for tuning.

    Args:
        hyperparams (Dict[str, Any]): Dictionary that contains tuning iteration configuration.
    """
    device = set_device()
    model = AMGCModel(num_classes=hyperparams["num_classes"])
    model.to(device)
    image_shape = hyperparams["image_width"], hyperparams["image_height"]
    basic_transforms = v2.Compose(
        [v2.Resize(image_shape), v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)]
    )
    dataset = datasets.ImageFolder(
        root=hyperparams["dataset_path"], transform=basic_transforms)
    total_size = len(dataset)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, _ = random_split(
        dataset, [train_size, val_size, test_size]
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
    valid_dataloader = DataLoader(
        val_dataset, batch_size=hyperparams["batch_size"], shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    learning_rate = hyperparams["learning_rate"]
    optimizer = get_optimizer(
        hyperparams["optimizer"], model.parameters(), learning_rate)
    early_stop_callback = EarlyStopping(
        hyperparams["patience"], hyperparams["early_stopping_metric"])
    callbacks = [early_stop_callback]
    trainer = Trainer(
        model,
        train_dataloader,
        valid_dataloader,
        criterion,
        optimizer,
        device,
        hyperparams["num_classes"],
        hyperparams["epochs"],
        callbacks=callbacks,
        tune=True
    )
    trainer.train()


def tune(config: Dict[str, Any]):
    """
    Function for running hyperopt tunning from dict configuration.

    Args:
        config (Dict[str, Any]): Dictionary with hyperparameters to optimize.
    """
    search_space = {}
    for key, value in config.items():
        if not isinstance(value, List):
            value = [value]
        search_space[key] = hp.choice(key, value)
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=100,
        trials=Trials()
    )
    print(best)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "-y", "--yaml", help="Path to yaml tune config file")
    args = args_parser.parse_args()
    config_params = load_config(args.yaml)
    tune(config_params)
