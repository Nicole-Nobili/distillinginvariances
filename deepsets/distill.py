# Run the distillation framework.
# Train a student network through the knowledge distillation framework.

import os
import time
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--config_file", type=str, default="default_cofig.yml")
parser.add_argument("--device", type=str)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

import random

random.seed(args.seed)

import numpy as np

np.random.seed(args.seed)

import torch

torch.manual_seed(args.seed)

import yaml
import torch.nn as nn
import torch_geometric
from torch.utils.data import DataLoader

from distiller import Distiller
import util


def main(config: dict):
    device = args.device
    outdir = util.make_output_directory("distilled_deepsets", config["outdir"])
    util.save_config_file(config, outdir)
    config_teacher = util.load_config_file(os.path.join(config["teacher"], "config.yml"))

    config["outdir"] = os.path.join(config["outdir"], f"seed{args.seed}")
    outdir = util.make_output_directory("distilled_deepsets", config["outdir"])

    train_data = util.import_data(device, config["data_hyperparams"], train=True)
    util.print_data_deets(train_data, "Training")
    valid_data = util.import_data(device, config["data_hyperparams"], train=False)
    util.print_data_deets(valid_data, "Validation")

    print(util.tcols.OKGREEN + "Teacher network" + util.tcols.ENDC)
    teacher_model = util.get_model(
        config_teacher["model_type"], config_teacher["model_hyperparams"]
    )
    weights_file = os.path.join(config["teacher"], "seed42", "model.pt")
    teacher_model.load_state_dict(torch.load(weights_file))
    util.profile_model(teacher_model, train_data, outdir)

    print(util.tcols.OKGREEN + "Student network" + util.tcols.ENDC)
    student_model = util.get_model(config["model_type"], config["model_hyperparams"])
    util.profile_model(student_model, train_data, outdir)

    distill_hyperparams = config["distill_hyperparams"]
    distill = Distiller(student_model, teacher_model, device, **distill_hyperparams)
    hist = distill.distill(train_data, valid_data)

    util.loss_plot(hist["student_train_losses"], hist["student_valid_losses"], outdir)
    util.accu_plot(hist["student_train_accurs"], hist["student_valid_accurs"], outdir)
    util.loss_plot(
        hist["distill_train_losses"],
        hist["distill_valid_losses"],
        outdir,
        "distillation"
    )


if __name__ == "__main__":
    with open(args.config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    main(config)
