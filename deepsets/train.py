# from the deepset paper: https://github.com/manzilzaheer/DeepSets/blob/master/PointClouds/classifier.py

import argparse
import time

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import util
from modelnet_data import Modelnet40DataLoader


def main(config: dict):
    device = util.define_torch_device()
    outdir = util.make_output_directory('trained_deepsets', config['outdir'])
    util.save_config_file(config, outdir)

    train_data = import_data(device, config, valid=False)
    valid_data = import_data(device, config, valid=True)

    input_dim = train_data.dataset[0]['pointcloud'].size()[-1]
    output_dim = len(train_data.dataset.classes)
    config['model_hyperparams'].update({'output_dim': output_dim})
    config['model_hyperparams'].update({'input_dim': input_dim})
    model = util.choose_deepsets(config['deepsets_type'], config['model_hyperparams'])

    hist = train(model, train_data, valid_data, device, config['training_hyperparams'])

    torch.save(model, outdir)


def import_data(device: str, config: dict, valid: bool) -> DataLoader:
    """Imports the data into a torch data set."""
    data_hyperparams = config['data_hyperparams']
    if valid:
        transforms = util.get_pointcloud_transformations(data_hyperparams['valid_trans'])
    else:
        transforms = util.get_pointcloud_transformations(data_hyperparams['train_trans'])
    data = Modelnet40DataLoader(data_hyperparams['pc_rootdir'], valid, transforms)
    inv_classes = {i: cat for cat, i in data.classes.items()}

    print(util.tcols.OKGREEN + "Data loaded: " + util.tcols.ENDC)
    if valid:
        data_hyperparams['torch_dataloader'].update({'batch_size': len(data)})
        print('Validation dataset size: ', len(data))
    else:
        print('Training dataset size: ', len(data))
    print('Number of classes: ', len(data.classes))
    print('Sample pointcloud shape: ', data[0]['pointcloud'].size())
    print('Class: ', inv_classes[data[0]['category']], '\n')

    dataloader_args = config['data_hyperparams']['torch_dataloader']
    if device == 'cpu':
        return torch.utils.data.DataLoader(data, **dataloader_args)

    return torch.utils.data.DataLoader(data, pin_memory=True, **dataloader_args)


def train(
    model: nn.Module,
    train_data: DataLoader,
    valid_data: DataLoader,
    device: str,
    training_hyperparams: dict
):
    """Trains a given model on given data for a number of epochs."""
    model = model.to(device)
    epochs = training_hyperparams['epochs']
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_hyperparams['lr'],
        weight_decay=training_hyperparams['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(range(400, epochs, 400)), gamma=0.1
    )
    print(util.tcols.OKGREEN + "Optimizer summary: " + util.tcols.ENDC)
    print(optimizer)

    loss_function = torch.nn.CrossEntropyLoss().cuda()

    print(util.tcols.OKCYAN + "\n\nTraining the deepsets model..." + util.tcols.ENDC)
    all_train_loss = []
    all_train_accu = []
    all_valid_loss = []
    all_valid_accu = []
    epochs_no_improve = 0
    epochs_es_limit = training_hyperparams['early_stopping']
    best_accu = 0

    model.train()
    for epoch in range(epochs):
        batch_loss_sum = 0
        totnum_batches = 0
        batch_accu_sum = 0
        for batch in train_data:
            x_batch = batch['pointcloud'].to(device).float()
            y_batch = batch['category'].to(device)
            y_pred = nn.functional.softmax(model(x_batch), dim=1)

            loss = loss_function(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            clip_grad(model, training_hyperparams['clip_grad'])
            optimizer.step()

            batch_loss_sum += loss
            totnum_batches += 1
            batch_accu_sum += torch.sum(y_pred.max(dim=1)[1] == y_batch)/len(y_batch)

        train_loss = batch_loss_sum/totnum_batches
        train_accu = batch_accu_sum/totnum_batches
        scheduler.step()

        x_valid = valid_data['pointcloud'].to(device).float()
        y_valid = valid_data['category'].to(device)
        y_pred = model.predict(x_valid)
        valid_loss = loss_function(y_pred, y_valid)
        valid_accu = torch.sum(y_pred == y_valid)/len(y_valid)
        if valid_accu <= best_accu:
            epochs_no_improve += 1
        else:
            best_accu = valid_accu
            epochs_no_improve = 0

        if early_stopping(epochs_no_improve, epochs_es_limit):
            break

        all_train_loss.append(train_loss.item())
        all_valid_loss.append(valid_loss.item())
        all_train_accu.append(train_accu)
        all_valid_accu.append(valid_accu)

        print_metrics(epoch, epochs, train_loss, train_accu, valid_loss, valid_accu)

    return {
        "train_losses": all_train_loss,
        "train_accurs": all_train_accu,
        "valid_losses": all_valid_loss,
        "valid_accurs": all_valid_accu
    }


def print_metrics(epoch, epochs, train_loss, train_accu, valid_loss, valid_accu):
    """Prints the training and validation metrics in a nice format."""
    print(
        util.tcols.OKGREEN + f"Epoch : {epoch + 1}/{epochs}\n" + util.tcols.ENDC +
        f"Train loss (average) = {train_loss.item():.8f}\n"
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
        total_norm += param_norm ** 2
    total_norm = total_norm ** (0.5)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            p.grad.data.mul_(clip_coef)
    return total_norm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config_file", type=str, default='default_cofig.yml')
    args = parser.parse_args()
    with open(args.config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    main(config)
