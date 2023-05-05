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



import torch
import copy
from datasets import load_datasets
from models import GAT, ResidualGAT
from train_test_utils import train, test, calculate_f1_score, evaluate_metrics

def GAT_implementation(data, device):
    model_gat = GAT(data=data, heads_layer1=8, heads_layer2=1, dropout=0.6, dropout_alphas=0.6).to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model_gat.parameters(), lr=0.005, weight_decay=0.0005)

    train_acc = []
    val_acc = []

    f1_before = calculate_f1_score(model_gat, data, data.test_mask)
    print("F1 score before training: {:.4f}".format(f1_before))

    best_val_acc = 0
    patience = 100
    counter = 0
    best_model = None

    for epoch in range(1, 200 + 1):
        train(model_gat, data, optimizer)
        if epoch % 10 == 0:
            acc_train, acc_val = test(model_gat, data)
            train_acc.append(acc_train)
            val_acc.append(acc_val)
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}'
            print(log.format(epoch, acc_train, acc_val))

            if acc_val > best_val_acc:
                best_val_acc = acc_val
                counter = 0
                best_model = copy.deepcopy(model_gat)
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping.")
                    break

    model_gat = best_model

    evaluate_metrics(model_gat, data)

    return model_gat

def Residual_GAT(data, device):
    model_gat = ResidualGAT(data=data, heads_layer1=10, heads_layer2=8, dropout=0.8, dropout_alphas=0.4).to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model_gat.parameters(), lr=0.002, weight_decay=6e-4)

    train_acc = []
    val_acc = []

    f1_before = calculate_f1_score(model_gat, data, data.test_mask)
    print("F1 score before training: {:.4f}".format(f1_before))

    best_val_acc = 0
    patience = 10
    counter = 0
    best_model = None

    for epoch in range(1, 200 + 1):
        train(model_gat, data, optimizer)
        if epoch % 10 == 0:
            acc_train, acc_val = test(model_gat, data)
            train_acc.append(acc_train)
            val_acc.append(acc_val)
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}'
            print(log.format(epoch, acc_train, acc_val))

            if acc_val > best_val_acc:
                best_val_acc = acc_val
                counter = 0
                best_model = copy.deepcopy(model_gat)
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping.")
                    break

    model_gat = best_model

    evaluate_metrics(model_gat, data)

    return model_gat

def Residual_GAT_cora(data, device):
    model_gat = ResidualGAT(data=data, heads_layer1=16, heads_layer2=8, dropout=0.7, dropout_alphas=0.7).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model_gat.parameters(), lr=0.001, weight_decay=0.001)

    train_acc = []
    val_acc = []

    f1_before = calculate_f1_score(model_gat, data, data.test_mask)
    print("F1 score before training: {:.4f}".format(f1_before))

    best_val_acc = 0
    patience = 10
    counter = 0
    best_model = None

    for epoch in range(1, 200 + 1):
        train(model_gat, data, optimizer)
        if epoch % 10 == 0:
            acc_train, acc_val = test(model_gat, data)
            train_acc.append(acc_train)
            val_acc.append(acc_val)
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}'
            print(log.format(epoch, acc_train, acc_val))

            if acc_val > best_val_acc:
                best_val_acc = acc_val
                counter = 0
                best_model = copy.deepcopy(model_gat)
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping.")
                    break

    model_gat = best_model

    evaluate_metrics(model_gat, data)

    return model_gat

def Residual_GAT_citeseer(data, device):
    model_gat = ResidualGAT(data=data, heads_layer1=16, heads_layer2=1, dropout=0.7, dropout_alphas=0.7).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model_gat.parameters(), lr=0.005, weight_decay=0.0005)

    train_acc = []
    val_acc = []

    f1_before = calculate_f1_score(model_gat, data, data.test_mask)
    print("F1 score before training: {:.4f}".format(f1_before))

    best_val_acc = 0
    patience = 10
    counter = 0
    best_model = None

    for epoch in range(1, 200 + 1):
        train(model_gat, data, optimizer)
        if epoch % 10 == 0:
            acc_train, acc_val = test(model_gat, data)
            train_acc.append(acc_train)
            val_acc.append(acc_val)
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}'
            print(log.format(epoch, acc_train, acc_val))

            if acc_val > best_val_acc:
                best_val_acc = acc_val
                counter = 0
                best_model = copy.deepcopy(model_gat)
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping.")
                    break

    model_gat = best_model

    evaluate_metrics(model_gat, data)

    return model_gat

def Residual_GAT_pubmed(data, device):
    model_gat = ResidualGAT(data=data, heads_layer1=6, heads_layer2=6, dropout=0.8, dropout_alphas=0.2).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model_gat.parameters(), lr=0.001, weight_decay=6e-4)

    train_acc = []
    val_acc = []

    f1_before = calculate_f1_score(model_gat, data, data.test_mask)
    print("F1 score before training: {:.4f}".format(f1_before))

    best_val_acc = 0
    patience = 10
    counter = 0
    best_model = None

    for epoch in range(1, 200 + 1):
        train(model_gat, data, optimizer)
        if epoch % 10 == 0:
            acc_train, acc_val = test(model_gat, data)
            train_acc.append(acc_train)
            val_acc.append(acc_val)
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}'
            print(log.format(epoch, acc_train, acc_val))

            if acc_val > best_val_acc:
                best_val_acc = acc_val
                counter = 0
                best_model = copy.deepcopy(model_gat)
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping.")
                    break

    model_gat = best_model

    evaluate_metrics(model_gat, data)

    return model_gat


def main():
    data_cora, data_citeseer, data_pubmed = load_datasets()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Training and Evaluating Cora on GAT")
    print("------------------------------------")
    model_cora_gat = GAT_implementation(data_cora, device)

    print("\nTraining and Evaluating Pubmed on GAT")
    print("------------------------------------")
    model_pubmed_gat = GAT_implementation(data_pubmed, device)

    print("\nTraining and Evaluating Citeseer on GAT")
    print("------------------------------------")
    model_citeseer_gat = GAT_implementation(data_citeseer, device)

    print("\nTraining and Evaluating Cora on Residual GAT")
    print("------------------------------------")
    model_cora_gat = Residual_GAT_cora(data_cora, device)

    print("\nTraining and Evaluating Pubmed on Residual GAT")
    print("------------------------------------")
    model_pubmed_gat = Residual_GAT_pubmed(data_pubmed, device)

    print("\nTraining and Evaluating Citeseer on Residual GAT")
    print("------------------------------------")
    model_citeseer_gat = Residual_GAT_citeseer(data_citeseer, device)

if __name__ == "__main__":
    main()
