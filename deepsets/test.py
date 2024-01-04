# Validation script for the deepsets network, computing all the metrics of interest.

import os
import time
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--models_dir", type=str)
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
import torchmetrics

import util


def main(args: dict):
    device = util.define_torch_device()
    model_dirs = [x[0] for x in os.walk(args.models_dir)][1:]
    config = util.load_config_file(os.path.join(args.models_dir, "config.yml"))
    valid_data = util.import_data(device, config["data_hyperparams"], train=False)
    util.print_data_deets(valid_data, "Validation")

    model = util.get_model(config)
    util.profile_model(model, valid_data, args.models_dir)
    all_metrics = {"accu": [], "nlll": [], "ecel": []}

    if "teacher" in config.keys():
        all_metrics.update({"top1_agreement": [], "teach_stu_kldiv": []})
        config_teacher = util.load_config_file(
            os.path.join(config["teacher"], "config.yml")
        )
        print(util.tcols.OKGREEN + "Teacher network" + util.tcols.ENDC)
        teacher_model = util.get_model(config_teacher)
        weights_file = os.path.join(config["teacher"], "seed42", "model.pt")
        teacher_model.load_state_dict(torch.load(weights_file))
        util.profile_model(teacher_model, train_data, outdir)

    for model_dir in model_dirs:
        print(util.tcols.HEADER + f"Model at: {model_dir}" + util.tcols.ENDC)
        weights_file = os.path.join(model_dir, "model.pt")
        metrics = validate(model, weights_file, valid_data, device)
        if "teacher" in config.keys():
            metrics.update(compute_fidelity(model, teacher_model, valid_data, device))
        for metric, value in metrics.items():
            all_metrics[metric].append(value)

    print(util.tcols.OKGREEN + "Average model metrics: " + util.tcols.ENDC)
    metrics_file_path = os.path.join(args.models_dir, "metrics_avg.log")
    with open(metrics_file_path, "a") as metrics_file:
        for metric, value in all_metrics.items():
            metric_mean = np.mean(value)
            metric_std = np.std(value)
            print(f"{metric}: {metric_mean:.3f} ± {metric_std:.3f}")
            metrics_file.write(f"{metric_mean:.3f} ± {metric_std:.3f}")



    print(util.tcols.OKCYAN + "Wrote average metrics to file in: " + util.tcols.ENDC)
    print(f"{metrics_file_path}")



def validate(model: nn.Module, weights_file: str, valid_data: DataLoader, device: str):
    """Run the model on the test data and save all relevant metrics to file."""
    model.load_state_dict(torch.load(weights_file))
    model.to(device)
    nll = nn.NLLLoss().cuda()
    ece = torchmetrics.classification.MulticlassCalibrationError(num_classes=40)

    batch_accu_sum = 0
    batch_nlll_sum = 0
    batch_ecel_sum = 0
    totnum_batches = 0
    for data in valid_data:
        data = data.to(device)
        y_true = data.y.flatten()
        y_pred = model.predict(data.pos)

        accu = torch.sum(y_pred.max(dim=1)[1] == y_true) / len(y_true)

        log_probs = nn.LogSoftmax(dim=1)(y_pred)
        nll_loss = nll(log_probs, y_true)
        ece_loss = ece(y_pred, y_true)
        inv_loss = test_perm_inv(model, data)

        batch_accu_sum += accu
        batch_nlll_sum += nll_loss
        batch_ecel_sum += ece_loss
        batch_invl_sum += inv_loss
        totnum_batches += 1

    metrics = {
        "accu": (batch_accu_sum / totnum_batches).cpu().item(),
        "nlll": (batch_nlll_sum / totnum_batches).cpu().item(),
        "ecel": (batch_ecel_sum / totnum_batches).cpu().item(),
        "invl": (batch_invl_sum / totnum_batches).cpu().item(),
    }
    print_metrics(metrics)

    return metrics


def compute_fidelity(student, teacher, valid_data: DataLoader, device: str):
    """Compute the student-teacher average top-1 agreement and the KL divergence."""
    kldiv = nn.KLDivLoss(reduction="batchmean", log_target=True)
    top1_agreement_running = []
    kldiv_loss_running = []

    teacher.to(device)
    student.to(device)
    for data in valid_data:
        data = data.to(device)
        y_teacher = teacher.predict(data.pos)
        y_student = student.predict(data.pos)
        top1_agreement = torch.sum(
                y_student.max(dim=1)[1] == y_teacher.max(dim=1)[1]
            ) / len(y_student)
        kldiv_loss = kldiv(
                nn.functional.log_softmax(y_teacher, dim=1),
                nn.functional.log_softmax(y_student, dim=1)
            )
        top1_agreement_running.append(top1_agreement.cpu().item())
        kldiv_loss_running.append(kldiv_loss.cpu().item())

    valid_top1_agreement = np.mean(top1_agreement_running)
    valid_kldiv_loss = np.mean(kldiv_loss_running)

    return {
        "top1_agreement": valid_top1_agreement,
        "teach_stu_kldiv": valid_kldiv_loss
    }


def invariance_measure(y_normal: torch.Tensor, y_transf: torch.Tensor):
    """Compute the difference of a model output given normal and symmetry trans data."""
    y_normal = nn.functional.softmax(y_normal, dim=1)
    y_transf = nn.functional.softmax(y_transf, dim=1)

    return torch.sum(torch.norm(y_normal - y_transf, dim=1))


def test_perm_inv(model: nn.Module, data: DataLoader):
    """Computes how invariant a model is with respect to a permutation of the data."""
    permutation = torch.randperm(data.pos.size(1))
    data_permuted = data.pos[:, permutation]

    y_normal = model.predict(data.pos)
    y_transf = model.predict(data_permuted)

    inv_measures = invariance_measure(y_normal, y_transf)
    return torch.sum(torch.cat(inv_measures))


def print_metrics(metrics: dict):
    """Prints the training and validation metrics in a nice format."""
    for key, value in metrics.items():
        print(f"{key}: {value:.8f}")
    print("")


if __name__ == "__main__":
    main(args)
