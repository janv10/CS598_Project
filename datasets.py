# Jahnvi Patel (jpate201@illinois.edu)
# Reference: Graph Attention Networks by Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio
# Link to the paper: https://arxiv.org/abs/1710.10903
# Link to the source code: https://github.com/PetarV-/GAT

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

def load_datasets():
    dataset_cora = Planetoid(root="./tmp", name="Cora", transform=T.NormalizeFeatures())
    dataset_citeseer = Planetoid(root="./tmp", name="CiteSeer", transform=T.NormalizeFeatures())
    dataset_pubmed = Planetoid(root="./tmp", name="Pubmed", transform=T.NormalizeFeatures())

    data_cora = dataset_cora[0]
    data_citeseer = dataset_citeseer[0]
    data_pubmed = dataset_pubmed[0]

    return data_cora, data_citeseer, data_pubmed
