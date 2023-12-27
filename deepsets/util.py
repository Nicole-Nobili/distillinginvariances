# Utility methods to support trivial tasks done in other files.

import os
import warnings
import subprocess

import yaml
import torch
import torch.nn as nn
import torchvision
import torchinfo

from deepsets import DeepSetsEquivariant
from deepsets import DeepSetsInvariant
import modelnet_transforms


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


def get_pointcloud_transformations(transformations: dict):
    """Gets the transformation composition from the config."""
    transforms = []
    if transformations["sampling"]:
        transforms.append(modelnet_transforms.PointSampler(transformations["sampling"]))
    if transformations["norm"]:
        transforms.append(modelnet_transforms.Normalise())
    if transformations["rotation"]:
        transforms.append(modelnet_transforms.RandRotationZ())
    if transformations["noise"]:
        transforms.append(modelnet_transforms.RandomNoise())
    if transformations["tensor"]:
        transforms.append(modelnet_transforms.ToTensor())

    return torchvision.transforms.Compose(transforms)


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


def get_free_gpu(threshold_vram_usage=3000, max_gpus=1):
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
