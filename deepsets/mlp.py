# MLP model classes.

import numpy as np

import torch
import torch.nn as nn


def activ_string_to_torch(activ: str):
    """Converts activation name from string to a torch objects."""
    activations = {
        "relu": lambda: nn.ReLU(inplace=True),
        "tanh": lambda: nn.Tanh(),
        "sigmoid": lambda: nn.Sigmoid(),
    }
    activation = activations.get(activ, lambda: None)()
    if activation is None:
        raise ValueError(
            f"Activation {activation} not implemented! Go to deepsets.py and add it."
        )

    return activation


class MLPBasic(nn.Module):
    """Basic MLP with variable number of layers."""

    def __init__(self, input_dim: int, layers: list, output_dim: int, activ: str):
        super(MLPBasic, self).__init__()
        self.activ = activ
        self.input_dim = input_dim
        self.layers = layers
        self.output_dim = output_dim

        self._construct_mlp()

    def _construct_mlp(self):
        self.layers.insert(0, self.input_dim)
        self.layers.append(self.output_dim)

        self.mlp = nn.Sequential()
        for nlayer in range(len(self.layers)):
            self.mlp.append(nn.Linear(self.layers[nlayer], self.layers[nlayer + 1]))
            if nlayer == len(self.layers) - 2:
                break

            activation = activ_string_to_torch(self.activ)
            self.mlp.append(activation)

    def forward(self, x):
        return self.mlp(x)

    def predict(self, x):
        self.eval()
        return self.mlp(x)
