# Tests the dataloaders to check if the data is imported correctly.

import sys

sys.path.append("../")

from dataloader_mnist import MnistDataloader
from dataloader_modelnet import Modelnet40DataLoader

import torchvision
import modelnet_transforms


def test_mnist_data_loading():
    images_filepath = "./data/MNIST/t10k-images.idx3-ubyte"
    labels_filepath = "./data/MNIST/t10k-labels.idx1-ubyte"
    train_dataloader = MnistDataloader(images_filepath, labels_filepath)
    train_data = train_dataloader.load_images_labels_raw()
    train_data.show_image(100)


def test_modelnet_data_loading():
    pointcloud_filepath = "./data/ModelNet40/"
    pointcloud_data = Modelnet40DataLoader(
        pointcloud_filepath, validation=False, transforms=None
    )
    pointcloud_data.show_pointcloud(0)


def test_modelnet_transformed_data_loading():
    pointcloud_filepath = "./data/ModelNet40/"
    transforms = torchvision.transforms.Compose(
        [
            modelnet_transforms.PointSampler(1024),
            modelnet_transforms.Normalise(),
            modelnet_transforms.RandRotationZ(),
            modelnet_transforms.RandomNoise(),
            modelnet_transforms.ToTensor(),
        ]
    )
    pointcloud_data = Modelnet40DataLoader(
        pointcloud_filepath, validation=False, transforms=transforms
    )
    pointcloud_data.show_pointcloud(0, "modelnet_images_transformed")


if __name__ == "__main__":
    test_mnist_data_loading()
    test_modelnet_data_loading()
    test_modelnet_transformed_data_loading()
