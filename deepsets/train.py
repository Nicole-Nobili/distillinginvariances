# Training script for the deepsets network.

import os
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--config_file", type=str)
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
import deepspeed
import torch.nn as nn
import torch_geometric
from torch.utils.data import DataLoader

import util



def main(config):
    device = args.device
    outdir = util.make_output_directory("trained_models", config["outdir"])
    util.save_config_file(config, outdir)

    config["outdir"] = os.path.join(config["outdir"], f"seed{args.seed}")
    outdir = util.make_output_directory("trained_models", config["outdir"])

    train_data = util.import_data(device, config["data_hyperparams"], train=True)
    util.print_data_deets(train_data, "Training")
    valid_data = util.import_data(device, config["data_hyperparams"], train=False)
    util.print_data_deets(valid_data, "Validation")

    model = util.get_model(config["model_type"], config["model_hyperparams"])
    hist = train(model, train_data, valid_data, device, config["training_hyperparams"])

    model_file = os.path.join(outdir, "model.pt")
    torch.save(model.state_dict(), model_file)
    util.loss_plot(hist["train_losses"], hist["valid_losses"], outdir)
    util.accu_plot(hist["train_accurs"], hist["valid_accurs"], outdir)
    

def train(
    model: nn.Module,
    train_data: DataLoader,
    valid_data: DataLoader,
    device: str,
    training_hyperparams: dict,
):
    """Trains a given model on given data for a number of epochs."""
    model = model.to(device)
    epochs = training_hyperparams["epochs"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_hyperparams["lr"],
        weight_decay=training_hyperparams["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=training_hyperparams["lr_patience"],
            factor=0.1,
            threshold=1e-3,
            verbose=True
        )
    print(util.tcols.OKGREEN + "Optimizer summary: " + util.tcols.ENDC)
    print(optimizer)

    loss_function = util.choose_loss(training_hyperparams["loss"], device)

    print(util.tcols.OKCYAN + "\n\nTraining model..." + util.tcols.ENDC)
    all_train_loss = []
    all_train_accu = []
    all_valid_loss = []
    all_valid_accu = []
    epochs_no_improve = 0
    epochs_es_limit = training_hyperparams["early_stopping"]
    best_accu = 0

    prof = deepspeed.profiling.flops_profiler.FlopsProfiler(model)
    profile_epoch = 0
    for epoch in range(epochs):
        if epoch == profile_epoch:
            prof.start_profile()

        batch_loss_sum = 0
        batch_accu_sum = 0
        totnum_batches = 0
        model.train()
        for data in train_data:
            data = data.to(device)

            if training_hyperparams["translate"]:
                data = translate_data(data)
            y_pred = model(data)
            y_true = data.y.flatten()

            if training_hyperparams["loss"] == "nll":
                y_pred = nn.functional.log_softmax(y_pred, dim=1)
            loss = loss_function(y_pred, y_true)
            optimizer.zero_grad()
            loss.backward()
            if "clip_grad" in training_hyperparams.keys():
                clip_grad(model, training_hyperparams["clip_grad"])
            optimizer.step()

            batch_loss_sum += loss
            totnum_batches += 1
            batch_accu_sum += torch.sum(y_pred.max(dim=1)[1] == y_true) / len(y_true)

            if epoch == profile_epoch and totnum_batches == 1:
                prof.stop_profile()
                print(util.tcols.OKGREEN + "Model flops: " + util.tcols.ENDC, end="")
                print(f"{prof.get_total_flops(as_string=True)}")

                prof.end_profile()
                print("")

        train_loss = batch_loss_sum / totnum_batches
        train_accu = batch_accu_sum / totnum_batches

        batch_loss_sum = 0
        batch_accu_sum = 0
        totnum_batches = 0
        for data in valid_data:
            data = data.to(device)
            y_true = data.y.flatten()
            y_pred = model.predict(data)

            if training_hyperparams["loss"] == "nll":
                y_pred = nn.functional.log_softmax(y_pred, dim=1)
            loss = loss_function(y_pred, y_true)
            accu = torch.sum(y_pred.max(dim=1)[1] == y_true) / len(y_true)
            batch_accu_sum += accu
            batch_loss_sum += loss
            totnum_batches += 1

        valid_loss = batch_loss_sum / totnum_batches
        valid_accu = batch_accu_sum / totnum_batches

        if round(valid_accu.cpu().item(), 4) <= round(best_accu, 4):
            epochs_no_improve += 1
        else:
            best_accu = valid_accu.cpu().item()
            epochs_no_improve = 0

        scheduler.step(best_accu)
        if epochs_no_improve >= epochs_es_limit:
            break

        all_train_loss.append(train_loss.item())
        all_valid_loss.append(valid_loss.item())
        all_train_accu.append(train_accu.item())
        all_valid_accu.append(valid_accu.item())

        print_metrics(epoch, epochs, train_loss, train_accu, valid_loss, valid_accu)

    return {
        "train_losses": all_train_loss,
        "train_accurs": all_train_accu,
        "valid_losses": all_valid_loss,
        "valid_accurs": all_valid_accu,
    }


def translate_data(data: DataLoader):
    """Applies a random translation to each data point."""
    translation = torch.rand(data.batch_size, data.pos.size(-1)).to(data.pos.device)*0.1
    translation = translation.repeat_interleave(int(data.size(0)/data.batch_size), dim=0)
    data.pos = torch.add(data.pos, translation)

    return data


def print_metrics(epoch, epochs, train_loss, train_accu, valid_loss, valid_accu):
    """Prints the training and validation metrics in a nice format."""
    print(
        util.tcols.OKGREEN
        + f"Epoch : {epoch + 1}/{epochs}\n"
        + util.tcols.ENDC
        + f"Train loss (average) = {train_loss.item():.8f}\n"
        f"Train accuracy  = {train_accu:.8f}\n"
        f"Valid loss = {valid_loss.item():.8f}\n"
        f"Valid accuracy = {valid_accu:.8f}\n"
    )


def clip_grad(model, max_norm):
    """Clips the gradient for the backwards propagation."""
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm**2
    total_norm = total_norm ** (0.5)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            p.grad.data.mul_(clip_coef)
    return total_norm


if __name__ == "__main__":
    with open(args.config_file, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
            
    main(config)
