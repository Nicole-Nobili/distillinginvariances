# Graph convolution networks.

import numpy as np

import torch
import torch.nn as nn
import torch_geometric


class DGCNN(nn.Module):
    def __init__(self, output_dim, k=20, aggr='max'):
        super().__init__()

        self.conv1 = torch_geometric.nn.DynamicEdgeConv(
            torch_geometric.nn.MLP([2 * 3, 64, 64, 64]), k, aggr
        )
        self.conv2 = torch_geometric.nn.DynamicEdgeConv(
            torch_geometric.nn.MLP([2 * 64, 128]), k, aggr
        )
        self.lin1 = nn.Linear(128 + 64, 1024)

        self.mlp = torch_geometric.nn.MLP(
            [1024, 512, 256, output_dim], dropout=0.5, norm=None
        )

    def forward(self, data):
        x, batch = data.pos, data.batch
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = torch_geometric.nn.global_max_pool(out, batch)
        out = self.mlp(out)

        return out

    @torch.no_grad()
    def predict(self, data):
        self.eval()
        x, batch = data.pos, data.batch
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = torch_geometric.nn.global_max_pool(out, batch)
        out = self.mlp(out)

        return out
