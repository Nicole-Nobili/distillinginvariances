# Utility methods to support trivial tasks done in other files.

import os
import warnings
import subprocess
import random

import yaml
import numpy as np
import torch
import torch.nn as nn
import torchinfo
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import data
from deepsets import DeepSetsEquivariant
from deepsets import DeepSetsInvariant
from mlp import MLPBasic
from mlp import MLPReged


def make_output_directories(locations: list | str, outdir: str) -> list:
    """Create an output directory in a list of locations."""
    if isinstance(locations, str):
        return make_output_directory(locations, outdir)

    return [make_output_directory(location, outdir) for location in locations]


def make_output_directory(location: str, outdir: str) -> str:
    """Create the output directory in a designated location."""
    outdir = os.path.join(location, outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    return outdir


def choose_deepsets(choice: str, model_hyperparams: dict):
    """Imports a DeepSets model."""
    deepsets = {
        "ds_invariant": lambda: DeepSetsInvariant(**model_hyperparams),
        "ds_equivariant": lambda: DeepSetsEquivariant(**model_hyperparams),
    }
    model = deepsets.get(choice, lambda: None)()
    if model is None:
        raise ValueError(
            f"{choice} is not the name of a DS model. Options: {deepsets.keys()}."
        )

    print(tcols.OKBLUE + "Network architecture:" + tcols.ENDC)
    torchinfo.summary(model)

    return model


def choose_mlp(choice: str, model_hyperparams: dict):
    """Imports an MLP model."""
    mlp = {
        "mlp_basic": lambda: MLPBasic(**model_hyperparams),
        "mlp_reged": lambda: MLPReged(**model_hyperparams),
    }

    model = mlp.get(choice, lambda: None)()
    if model is None:
        raise ValueError(f"{choice} is not an MLP model. Options: {mlp.keys()}.")

    print(tcols.OKBLUE + "Network architecture:" + tcols.ENDC)
    torchinfo.summary(model)

    return model


def get_model(model_type: str, model_hyperparams: dict):
    """Get a model specified through a string in the configuration file."""
    if "ds" in model_type:
        model = choose_deepsets(model_type, model_hyperparams)
    elif "mlp" in model_type:
        model = choose_mlp(model_type, model_hyperparams)
    else:
        raise ValueError("Please specify in config  which kind of ML model you want.")

    return model


def choose_loss(choice, device):
    """Get a pytorch loss object given a string specified in the configuration file."""
    losses = {
        "ce": lambda: nn.CrossEntropyLoss().to(device),
        "nll": lambda: nn.NLLLoss().to(device)
    }

    loss = losses.get(choice, lambda: None)()
    if loss is None:
        raise ValueError(f"Loss {choice} not specified. Go to util.py and add it.")

    return loss


def import_data(device: str, config: dict, train: bool):
    """Imports the jet data and casts it into a torch DataLoader object."""
    print(tcols.OKGREEN + "Importing data... " + tcols.ENDC, end="")
    jet_data = data.HLS4MLData150(
        config["root"],
        config["nconst"],
        config["feats"],
        config["norm"],
        train,
    )
    if train:
        print("training data imported!")
    else:
        config["torch_dataloader"]["shuffle"] = False
        print("validation data imported!")

    dl_args = config["torch_dataloader"]
    tensor_dataset = jet_data.get_torch_dataset()
    if device == "cpu":
        return torch.utils.data.DataLoader(tensor_dataset, **dl_args)

    return torch.utils.data.DataLoader(tensor_dataset, pin_memory=True, **dl_args)


def print_data_deets(data, data_type: str):
    """Prints some key details of the data set, for sanity check at start of training."""
    x, y = next(iter(data))
    print(tcols.HEADER + f"{data_type} data details:" + tcols.ENDC)
    print(f"Dataset size: {len(data.dataset)}")
    print(f"Batched size: {x.size(0)}")
    print(f"Number of constituents: {x.size(1)}")
    print(f"Number of features: {x.size(2)}")
    print("")


def save_config_file(config: dict, outdir: str):
    """Saves the config file into given output directory."""
    outfile = os.path.join(outdir, "config.yml")
    with open(outfile, "w") as file:
        yaml.dump(config, file)


def load_config_file(config_file: str):
    """Loads a config file given a certain path."""
    with open(config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return config


def loss_plot(all_train_losses: list, all_valid_losses: list, outdir: str, label = ""):
    """Plots the loss for each epoch for the training and validation data."""
    epochs = list(range(len(all_train_losses)))
    plt.plot(
        epochs,
        all_train_losses,
        color="gray",
        label=f"Training loss",
    )
    plt.plot(epochs, all_valid_losses, color="navy", label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel(f"{label} loss")

    best_valid_loss = min(all_valid_losses)
    plt.text(
        np.min(epochs),
        np.max(all_train_losses),
        f"Min: {best_valid_loss:.3f}",
        verticalalignment="top",
        horizontalalignment="left",
        color="blue",
        fontsize=15,
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
    )
    plt.legend()
    plt.savefig(os.path.join(outdir, f"{label}_loss_epochs.pdf"))
    plt.close()
    print(tcols.OKGREEN + f"Loss vs epochs plot saved to {outdir}." + tcols.ENDC)


def accu_plot(all_train_accs: list, all_valid_accs: list, outdir: str):
    """Plots the loss for each epoch for the training and validation data."""
    epochs = list(range(len(all_train_accs)))
    plt.plot(
        epochs,
        all_train_accs,
        color="gray",
        label="Training Accuracy (average)",
    )
    plt.plot(epochs, all_valid_accs, color="navy", label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    best_valid_acc = max(all_valid_accs)
    plt.text(
        np.min(epochs),
        np.max(all_train_accs),
        f"Min: {best_valid_acc:.3f}",
        verticalalignment="top",
        horizontalalignment="left",
        color="blue",
        fontsize=15,
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
    )
    plt.legend()
    plt.savefig(os.path.join(outdir, "accu_epochs.pdf"))
    plt.close()
    print(tcols.OKGREEN + f"Accuracy vs epochs plot saved to {outdir}." + tcols.ENDC)


class tcols:
    """Pretty terminal colors ooooo."""
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
