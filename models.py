# Jahnvi Patel (jpate201@illinois.edu)
# Reference: Graph Attention Networks by Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio
# Link to the paper: https://arxiv.org/abs/1710.10903
# Link to the source code: https://github.com/PetarV-/GAT


import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import LayerNorm

class GAT(torch.nn.Module):
    def __init__(self, data, heads_layer1, heads_layer2, dropout, dropout_alphas):
        super().__init__()

        self.dropout = dropout
        num_features = data.num_features
        num_classes = len(data.y.unique())

        self.conv1 = GATConv(in_channels=num_features, out_channels=8,
                             heads=heads_layer1, concat=True, negative_slope=0.2, 
                             dropout=dropout_alphas)

        self.conv2 = GATConv(in_channels=8 * heads_layer1, out_channels=num_classes, 
                             heads=heads_layer2, concat=False, negative_slope=0.2,
                             dropout=dropout_alphas)

    def forward(self, data):
        x = data.x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, data.edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, data.edge_index)

        return F.log_softmax(x, dim=1)


class ResidualGAT(torch.nn.Module):
    def __init__(self, data, heads_layer1, heads_layer2, dropout, dropout_alphas):
        super().__init__()

        self.dropout = dropout
        num_features = data.num_features
        num_classes = len(data.y.unique())

        self.conv1 = GATConv(in_channels=num_features, out_channels=8,
                             heads=heads_layer1, concat=True, negative_slope=0.2, 
                             dropout=dropout_alphas)

        self.res_conv1 = GATConv(in_channels=num_features, out_channels=8,
                                 heads=heads_layer1, concat=True, negative_slope=0.2, 
                                 dropout=dropout_alphas)

        self.conv2 = GATConv(in_channels=8 * heads_layer1, out_channels=num_classes, 
                             heads=heads_layer2, concat=False, negative_slope=0.2,
                             dropout=dropout_alphas)

        self.res_conv2 = GATConv(in_channels=8 * heads_layer1, out_channels=num_classes, 
                                 heads=heads_layer2, concat=False, negative_slope=0.2,
                                 dropout=dropout_alphas)

    def forward(self, data):
        x = data.x
        x = F.dropout(x, p=self.dropout, training=self.training)

        x1 = self.conv1(x, data.edge_index)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        x1_res = self.res_conv1(x, data.edge_index)
        x1_res = F.elu(x1_res)

        x1 = x1 + x1_res

        x2 = self.conv2(x1, data.edge_index)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x2_res = self.res_conv2(x1, data.edge_index)

        x2 = x2 + x2_res

        return F.log_softmax(x2, dim=1)

