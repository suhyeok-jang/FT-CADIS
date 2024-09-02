# this file is based on code publicly available at
# https://github.com/alinlab/smoothing-catrs
# written by Jeong et al.

import os
from typing import *

import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
os.environ["IMAGENET_LOC_ENV"] = "IMAGENET_DIR"
CIFAR10_LOC = "./data/cifar10"

DATASETS = ["cifar10", "imagenet"]


def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "cifar10":
        return _cifar10(split)
    elif dataset == "imagenet":
        return _imagenet(split)


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset."""
    if dataset == "cifar10":
        return 10
    elif dataset == "imagenet":
        return 1000


def _cifar10(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR10(
            CIFAR10_LOC,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
        )

    elif split == "test":
        return datasets.CIFAR10(CIFAR10_LOC, train=False, download=True, transform=transforms.ToTensor())


def _imagenet(split: str) -> Dataset:
    if "IMAGENET_LOC_ENV" not in os.environ:
        raise RuntimeError("Environment variable for ImageNet directory not set")

    dir = os.environ["IMAGENET_LOC_ENV"]
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    return datasets.ImageFolder(subdir, transform)
