"""Module containing dataset classes."""
import os
from typing import Tuple

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelBinarizer
from PIL import Image


def load_filenames_and_labels_gtzan(images_dir: str):
    """Loads filenames and labels for gtzan dataset.

    Args:
        images_dir (str): Directory where files are stored.

    Returns:
        List[Tuple[str, str]]: List of tuples containing filename and label.
    """
    genres = os.listdir(images_dir)
    data = []
    encoding_mapping = {label: idx for idx, label in enumerate(genres)}
    for genre in genres:
        genre_dir = os.path.join(images_dir, genre)
        filenames = os.listdir(genre_dir)
        for filename in filenames:
            path_to_filename = os.path.join(genre_dir, filename)
            data.append((path_to_filename, encoding_mapping[genre]))
    return data


class DatasetGTZAN(Dataset):
    """Class for loading gtzan dataset."""

    def __init__(self, filenames_and_labels: Tuple[str, str]) -> None:
        super().__init__()
        self.filepaths, self.labels = zip(*filenames_and_labels)
        lb = LabelBinarizer()
        self.labels = torch.from_numpy(lb.fit_transform(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.load_image(index)
        return image, label

    def load_image(self, idx: int):
        """Loads image.

        Args:
            idx (int): index of image to be loaded.

        Returns:
            torch.tensor: Loaded image.
        """
        image = Image.open(self.filepaths[idx])
        image = torch.from_numpy(image)
        return image
