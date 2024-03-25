"""Module that handles hyperparameter optimization."""
from typing import Dict, Any, List
import argparse
import time
import json

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from hyperopt import hp, fmin, tpe, Trials

from model.model import AMGCModel
from trainer.trainer import Trainer
from callbacks.early_stopping import EarlyStopping
from utils.param_by_name import get_optimizer, get_learning_rate_scheduler
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
        [v2.Resize(image_shape), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
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
    lr_scheduler = get_learning_rate_scheduler(hyperparams, optimizer)
    base_save_path = f"./results/tune/model_{int(time.time())}"
    early_stop_callback = EarlyStopping(
        hyperparams["patience"], hyperparams["early_stopping_metric"], base_save_path
    )
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
        lr_scheduler,
        callbacks=callbacks,
        tune=True
    )
    result_dict = trainer.train()
    config_save_path = f"{base_save_path}/config.json"
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(hyperparams, f, indent=4)
    return result_dict


def create_search_space(
    config: Dict[str, Any],
    search_space: Dict[str, Any],
    prefix=""
) -> Dict[str, Any]:
    """Creates search space.

    Args:
        config (Dict): Configuration to get params.
        search_space (Dict): search_space to be filled.
        prefix (str): Prefix for key name in search_space.
    Returns:
        Dict: Filled search_space.
    """
    for key, value in config.items():
        if isinstance(value, Dict):
            new_prefix = f"{prefix}_{key}" if prefix else key
            search_space = create_search_space(value, search_space, new_prefix)
            continue
        if not isinstance(value, List):
            value = [value]
        key_to_fill = f"{prefix}_{key}" if prefix else key
        search_space[key_to_fill] = hp.choice(key_to_fill, value)
    return search_space


def tune(config: Dict[str, Any]):
    """
    Function for running hyperopt tunning from dict configuration.

    Args:
        config (Dict[str, Any]): Dictionary with hyperparameters to optimize.
    """
    search_space = {}
    search_space = create_search_space(config, search_space)
    trials = Trials()
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=4,
        trials=trials
    )
    print(best)

    best_trial = trials.best_trial
    print(best_trial)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "-y", "--yaml", help="Path to yaml tune config file")
    args = args_parser.parse_args()
    config_params = load_config(args.yaml)
    tune(config_params)
