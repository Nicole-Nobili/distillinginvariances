# Training script for the deepsets network.

import os
import time
import argparse
import pprint


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--config_file", type=str, default="    ")
parser.add_argument("--seed", type=int, default=42)
# Add an argument for Optuna tuning
parser.add_argument("--optuna", action="store_true", help="Run Optuna hyperparameter tuning")

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

import util

import optuna



def main(config, hyperparams=None, trial=None):
    
    """Optimization of Validation Metric: Ensure that the validation_metric returned by the main function is the metric
    you intend to optimize.This metric should reflect the performance of your model effectively."""
    
    # Override specific hyperparameters in config with those from Optuna
    if hyperparams:
        config["training_hyperparams"].update(hyperparams)
        
    device = util.define_torch_device()
    outdir = util.make_output_directory("trained_deepsets", config["outdir"])
    util.save_config_file(config, outdir)

    config["outdir"] = os.path.join(config["outdir"], f"seed{args.seed}")
    outdir = util.make_output_directory("trained_deepsets", config["outdir"])

    train_data = util.import_data(device, config["data_hyperparams"], train=True)
    util.print_data_deets(train_data, "Training")
    valid_data = util.import_data(device, config["data_hyperparams"], train=False)
    util.print_data_deets(valid_data, "Validation")

    model = util.get_model(config)
    # util.profile_model(model, train_data, outdir)  # seems not been done yet
    
    
    # Pass the Optuna trial object to the train function
    hist = train(model, train_data, valid_data, device, config["training_hyperparams"], trial)
    if "validation_metric" not in hist:
        print("Error: validation_metric not found in training results.")
        return None
    

    model_file = os.path.join(outdir, "model.pt")
    torch.save(model.state_dict(), model_file)
    util.loss_plot(hist["train_losses"], hist["valid_losses"], outdir)
    util.accu_plot(hist["train_accurs"], hist["valid_accurs"], outdir)
    
    return hist


def train(
    model: nn.Module,
    train_data: DataLoader,
    valid_data: DataLoader,
    device: str,
    training_hyperparams: dict,
    trial=None  # Add trial as an optional parameter

):
    """Trains a given model on given data for a number of epochs."""
    model = model.to(device)
    epochs = training_hyperparams["epochs"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_hyperparams["lr"],
        weight_decay=training_hyperparams["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(range(400, epochs, 400)), gamma=0.1
    )
    print(util.tcols.OKGREEN + "Optimizer summary: " + util.tcols.ENDC)
    print(optimizer)

    loss_function = torch.nn.CrossEntropyLoss().cuda()

    print(util.tcols.OKCYAN + "\n\nTraining model..." + util.tcols.ENDC)
    all_train_loss = []
    all_train_accu = []
    all_valid_loss = []
    all_valid_accu = []
    epochs_no_improve = 0
    epochs_es_limit = training_hyperparams["early_stopping"]
    # best_accu = 0

    # added best validation loss
    best_valid_loss = float('inf')

    for epoch in range(epochs):
        batch_loss_sum = 0
        totnum_batches = 0
        batch_accu_sum = 0
        model.train()
        for data in train_data:
            data = data.to(device)
            y_pred = model(data.pos)
            y_true = data.y.flatten()

            loss = loss_function(y_pred, y_true)
            optimizer.zero_grad()
            loss.backward()
            clip_grad(model, training_hyperparams["clip_grad"])
            optimizer.step()

            batch_loss_sum += loss
            totnum_batches += 1
            batch_accu_sum += torch.sum(y_pred.max(dim=1)[1] == y_true) / len(y_true)

        train_loss = batch_loss_sum / totnum_batches
        train_accu = batch_accu_sum / totnum_batches
        scheduler.step()

        for data in valid_data:
            data = data.to(device)
            y_true = data.y.flatten()
            y_pred = model.predict(data.pos)

            loss = loss_function(y_pred, y_true)
            accu = torch.sum(y_pred.max(dim=1)[1] == y_true) / len(y_true)
            batch_accu_sum += accu
            batch_loss_sum += loss
            totnum_batches += 1

        valid_loss = batch_loss_sum / totnum_batches
        valid_accu = batch_accu_sum / totnum_batches

        # Update best validation loss and check for early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
        else:
            if early_stopping(epochs_no_improve, epochs_es_limit):
                break
            
         # Reporting to Optuna and checking for pruning
        if trial:
            trial.report(valid_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

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
        "validation_metric": max(all_valid_accu)  # Use max since higher is better for accuracy

    }


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


def early_stopping(epochs_no_improve: int, epochs_limit: int) -> bool:
    """Stops the training if there has been no improvement in the loss."""
    if epochs_no_improve >= epochs_limit:
        return 1
    return 0


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



def objective(trial, config):
    # Suggesting hyperparameters for training
    training_hyperparams = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2),
        "weight_decay": trial.suggest_loguniform("weight_decay", 1e-10, 1e-3),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    }
    config["training_hyperparams"].update(training_hyperparams)

    # Suggesting hyperparameters for the model
    model_hyperparams = {
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
        "activ": trial.suggest_categorical("activ", ["relu", "tanh", "leaky_relu", "sigmoid"])
    }
    num_layers = trial.suggest_int("num_layers", 2, 15)
    layers = [trial.suggest_int(f"nr_layer_{i}", 128, 1024, step=128) for i in range(num_layers)]
    model_hyperparams["layers"] = layers
    config["model_hyperparams"].update(model_hyperparams)

    try:
        result = main(config, trial=trial)
        print(f"Trial {trial.number} Results:")
        pprint.pprint({
                "Parameters": trial.params,
                "Validation Metric": result['validation_metric']
                })
        return result["validation_metric"]
    except optuna.TrialPruned as e:
        print(f"Trial {trial.number} pruned.")
        raise e





def run_optuna_study(config):
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, config), n_trials=1000)
    print("Best hyperparameters:", study.best_trial.params)

   




if __name__ == "__main__":
    # Load config
    with open(args.config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
     # Decide whether to run a regular training or Optuna study based on a condition or argument
    if args.optuna: 
        run_optuna_study(config)
    else:
        main(config)