# Jahnvi Patel (jpate201@illinois.edu)
# Reference: Graph Attention Networks by Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio
# Link to the paper: https://arxiv.org/abs/1710.10903
# Link to the source code: https://github.com/PetarV-/GAT


import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    log_softmax = model(data)
    labels = data.y
    nll_loss = F.nll_loss(log_softmax[data.train_mask], labels[data.train_mask])
    nll_loss.backward()
    optimizer.step()

def compute_accuracy(model, data, mask):
    model.eval()
    logprob = model(data)
    _, y_pred = logprob[mask].max(dim=1)
    y_true = data.y[mask]
    acc = y_pred.eq(y_true).sum() / mask.sum().float()
    return acc.item()

def test(model, data):
    acc_train = compute_accuracy(model, data, data.train_mask)
    acc_val = compute_accuracy(model, data, data.val_mask)
    return acc_train, acc_val

def calculate_f1_score(model, data, mask):
    model.eval()
    logprob = model(data)
    _, y_pred = logprob[mask].max(dim=1)
    y_true = data.y[mask]
    f1 = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
    return f1

def calculate_precision(model, data, mask):
    model.eval()
    logprob = model(data)
    _, y_pred = logprob[mask].max(dim=1)
    y_true = data.y[mask]
    precision = precision_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
    return precision

def calculate_recall(model, data, mask):
    model.eval()
    logprob = model(data)
    _, y_pred = logprob[mask].max(dim=1)
    y_true = data.y[mask]
    recall = recall_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
    return recall

def evaluate_metrics(model, data):
    train_mask = data.train_mask
    val_mask = data.val_mask

    acc_train = compute_accuracy(model, data, train_mask)
    acc_val = compute_accuracy(model, data, val_mask)

    f1_train = calculate_f1_score(model, data, train_mask)
    f1_val = calculate_f1_score(model, data, val_mask)

    precision_train = calculate_precision(model, data, train_mask)
    precision_val = calculate_precision(model, data, val_mask)

    recall_train = calculate_recall(model, data, train_mask)
    recall_val = calculate_recall(model, data, val_mask)

    print("Train metrics:")
    print("  Accuracy:", acc_train)
    print("  Precision:", precision_train)
    print("  Recall:", recall_train)
    print("  F1 score:", f1_train)

    print("\nValidation metrics:")
    print("  Accuracy:", acc_val)
    print("  Precision:", precision_val)
    print("  Recall:", recall_val)
    print("  F1 score:", f1_val)