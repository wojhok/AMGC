""" Main module for managing program flow."""
import logging
import argparse
import random

import torch
from sklearn.model_selection import train_test_split

from model.model import AMGCModel
from dataloaders.dataset import DatasetGTZAN, load_filenames_and_labels_gtzan


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



def main(images_dir: str) -> None:
    """Main function.

    Args:
        images_dir (str): Path to an directory containing images. 
    """
    device = set_device()
    model = AMGCModel(num_classes=10)
    model.to(device)
    data = load_filenames_and_labels_gtzan(images_dir)
    random.shuffle(data)
    train, temp = train_test_split(data, test_size=0.3)
    valid, test = train_test_split(temp, test_size=0.5)
    train_dataset = DatasetGTZAN(train)
    valid_dataset = DatasetGTZAN(valid)
    test_dataset = DatasetGTZAN(test)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "-id", "--images-dir", help="Path to root_dir of images."
    )
    args = argument_parser.parse_args()
    main(args.images_dir)
