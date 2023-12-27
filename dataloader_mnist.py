# Loads MNIST data. Available here: https://zenodo.org/records/10058130
# Auguments the MNIST data with jitters.


import os
from array import array
from os.path import join
import struct
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import torch

import util


@dataclass
class RawDataset:
    """Raw dataset."""

    images: np.ndarray
    labels: array

    def show_image(self, image_idx: int, outdir: str = "mnist_image_examples"):
        """Displays and saves an MNIST image from the data set to disk."""
        util.make_output_directories("./", outdir)
        fig = plt.figure
        plt.imshow(self.images[image_idx], cmap=plt.cm.gray)
        title_text = f"image [{image_idx}] = {self.labels[image_idx]}"
        plt.title(title_text, fontsize=15)
        plt.savefig(join(outdir, f"image_{image_idx}"))


class MnistDataloader(object):
    def __init__(self, images_filepath: str, labels_filepath: str, **kwargs):
        """Loads MNIST images and labels.

        Attributes:
            images_filepath: The filepath to the images datafile.
            labels_filepath: The filepath to the labels associated with the images.
            kwargs: Hyperparameters for the augmentation on the images, e.g., jitters.
        """
        self.images_filepath = images_filepath
        self.labels_filepath = labels_filepath

    def _read_labels(self, labels_filepath: str) -> array:
        """Reads the labels to MNIST images."""
        labels = []
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f"Magic number mismatch, expected 2049, got {magic}")
            labels = array("B", file.read())

        return labels

    def _read_images(self, images_filepath: str) -> list:
        """Reads MNIST images."""
        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f"Magic number mismatch, expected 2051, got {magic}")
            image_data = array("B", file.read())

        images = []
        for idx in range(size):
            images.append([0] * rows * cols)
        for idx in range(size):
            img = np.array(image_data[idx * rows * cols : (idx + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[idx][:] = img

        return images

    def load_images_labels_raw(self) -> RawDataset:
        """Reads MNIST images and labels pair."""
        images = self._read_images(self.images_filepath)
        labels = self._read_labels(self.labels_filepath)

        return RawDataset(images=images, labels=labels)

    def load_images_labels_pytorch(
        self, device: str, batch_size: int = 64, shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        """Reads MNIST images and labels pair into a pytorch dataloader object."""
        images = self._read_images(self.images_filepath)
        labels = self._read_labels(self.labels_filepath)

        data = self._make_torch_dataset(images, labels)
        if device == "cpu":
            pytorch_loader = torch.utils.data.DataLoader(
                data, batch_size=batch_size, shuffle=shuffle
            )
        elif device == "gpu":
            pytorch_loader = torch.utils.data.DataLoader(
                data, batch_size=batch_size, shuffle=shuffle, pin_memory=True
            )
        else:
            raise ValueError(f"Device {device} not recognised!")

        return pytorch_loader

    def _make_torch_dataset(self, images, labels):
        """Converts the imported data into pytorch tensors."""
        images = torch.Tensor(images)
        labels = torch.Tensor(labels)

        return torch.utils.data.TensorDataset(images, labels)
