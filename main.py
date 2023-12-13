""" Main module for managing program flow."""
import logging
import argparse
import os
import random

import torch
from sklearn.model_selection import train_test_split

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

def load_filenames_and_labels_gtzan(images_dir: str):
    genres = os.listdir(images_dir)
    data = []
    for genre in genres:
        genre_dir = os.path.join(images_dir, genre)
        filenames = os.listdir(genre_dir)
        for filename in filenames:
            path_to_filename = os.path.join(genre_dir, filename)
            data.append((path_to_filename, genre))
    return data


def main(images_dir: str) -> None:
    """Main function."""
    device = set_device()
    model = AMGCModel(num_classes=10)
    logging.info(" Model architecture: %s", model)
    model.to(device)
    data = load_filenames_and_labels_gtzan(images_dir)
    random.shuffle(data)

    train, temp = train_test_split(data, test_size=0.3)
    validation, test = train_test_split(temp, test_size=0.5)

    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("-id", "--images-dir", help="Path to root_dir of images.")
    args = argument_parser.parse_args()
    main(args.images_dir)
