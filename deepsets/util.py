# Utility methods to support trivial tasks done in other files.

import os
import warnings
import subprocess

import yaml
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchinfo
import torch_geometric
import matplotlib.pyplot as plt

from deepsets import DeepSetsEquivariant
from deepsets import DeepSetsInvariant
from mlp import MLPBasic
import modelnet_transforms
from torch_geometric.transforms import SamplePoints


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
        "invariant": lambda: DeepSetsInvariant(**model_hyperparams),
        "equivariant": lambda: DeepSetsEquivariant(**model_hyperparams),
    }
    model = deepsets.get(choice, lambda: None)()
    if model is None:
        raise ValueError(
            f"{choice} is not the name of a DS model. Options: {deepsets.keys()}."
        )

    print(tcols.OKBLUE + "Network ready for training:" + tcols.ENDC)
    torchinfo.summary(model)

    return model


def choose_mlp(choice: str, model_hyperparams: dict):
    """Imports a DeepSets model."""
    mlp = {
        "basic": lambda: MLPBasic(**model_hyperparams),
        # "regularised": lambda: MLPRegularised(**model_hyperparams), WIP
    }
    model = mlp.get(choice, lambda: None)()
    if model is None:
        raise ValueError(
            f"{choice} is not the name of an MLP  model. Options: {mlp.keys()}."
        )

    print(tcols.OKBLUE + "Network ready for training:" + tcols.ENDC)
    torchinfo.summary(model)

    return model


def get_torchgeometric_pretransforms(pretransforms: dict):
    """Get the pre-transformations to include in the modelnet loader class.

    The data is saved to disk with these transformations applied.
    """
    tg_pretransforms = []
    if pretransforms["norm"]:
        tg_pretransforms.append(torch_geometric.transforms.NormalizeScale())

    return torch_geometric.transforms.Compose(tg_pretransforms)


def get_torchgeometric_transforms(transforms: dict):
    """Get the transformations that are applied to modelnet pointcloud data.

    These transformations are applied as each pointcloud is loaded.
    """
    tg_transforms = []
    if "sampling" in transforms.keys():
        tg_transforms.append(
            torch_geometric.transforms.SamplePoints(transforms["sampling"])
        )

    return torch_geometric.transforms.Compose(tg_transforms)


def save_config_file(config: dict, outdir: str):
    """Saves the config file into given output directory."""
    outfile = os.path.join(outdir, "config.yml")
    with open(outfile, "w") as file:
        yaml.dump(config, file)


def define_torch_device() -> torch.device:
    # Use gpu for training if available. Alert the user if not and use cpu.
    print("\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        device = torch.device(
            "cuda:" + str(get_free_gpu()) if torch.cuda.is_available() else "cpu"
        )
        if len(w):
            print(tcols.WARNING + "GPU not available." + tcols.ENDC)

    print("\033[92mUsing device:\033[0m", device)
    return device


def get_free_gpu(threshold_vram_usage=30000, max_gpus=1):
    """
    Returns the free gpu numbers on your system.

    Tto replace x in the string 'cuda:x'. The freeness is determined based on how much
    memory is currently being used on a gpu.

    Args:
        threshold_vram_usage: A GPU is free if vram usage is below the threshold.
        max_gpus: Max GPUs is the maximum number of gpus to assign.
    """

    # Get the list of GPUs via nvidia-smi.
    smi_query_result = subprocess.check_output(
        "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
    )

    # Extract the usage information
    gpu_info = smi_query_result.decode("utf-8").split("\n")
    gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
    gpu_info = [int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info]
    # Keep gpus under threshold only.
    free_gpus = [str(i) for i, mem in enumerate(gpu_info) if mem < threshold_vram_usage]
    free_gpus = free_gpus[: min(max_gpus, len(free_gpus))]
    gpus_to_use = ",".join(free_gpus)

    if not gpus_to_use:
        raise RuntimeError(tcols.FAIL + "No free GPUs found." + tcols.ENDC)

    return gpus_to_use


def loss_plot(all_train_losses: list, all_valid_losses: list, outdir: str):
    """Plots the loss for each epoch for the training and validation data."""
    epochs = list(range(len(all_train_losses)))
    plt.plot(
        epochs,
        all_train_losses,
        color="gray",
        label="Training Loss (average)",
    )
    plt.plot(epochs, all_valid_losses, color="navy", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    best_valid_loss = min(all_valid_losses)
    plt.text(
        np.min(epochs),
        np.max(all_train_losses),
        f"Min: {best_valid_loss:.2e}",
        verticalalignment="top",
        horizontalalignment="left",
        color="blue",
        fontsize=15,
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
    )
    plt.legend()
    plt.savefig(os.path.join(outdir, "loss_epochs.pdf"))
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
        f"Min: {best_valid_acc:.2e}",
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
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
