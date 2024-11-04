from __future__ import annotations
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim as optim
import sys
sys.path.append("c:\\Users\\p51\\OneDrive\\Desktop\\Dubokoucenje1")
from lab0.data import *
from torch.optim.lr_scheduler import ExponentialLR


class PTDeep(nn.Module):
    def __init__(self, layers: List, activation_fun):
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
        """
        super().__init__()
        self.layers = layers
        self.weights = nn.ParameterList([
            nn.Parameter(
                torch.randn(size=(layers[i], layers[i + 1]), dtype=torch.float64, requires_grad=True)
            )
            for i in range(len(layers) - 1)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(
                torch.zeros(layers[i + 1], dtype=torch.float64, requires_grad=True)
            )
            for i in range(len(layers) - 1)
        ])

        self.activation_fun = None
        if activation_fun == "relu":
            self.activation_fun = torch.relu
        elif activation_fun == "sigmoid":
            self.activation_fun = torch.sigmoid
        elif activation_fun == "tanh":
            self.activation_fun = torch.tanh

    def forward(self, X):
        h, scores = X, None
        for i in range(len(self.weights)):
            scores = h @ self.weights[i] + self.biases[i]
            h = self.activation_fun(scores)

        probs = torch.softmax(scores, dim=1)
        return probs

    def get_loss(self, X, Yoh_, lambd):
        Y = self.forward(X)
        loss = -torch.mean(torch.sum(Yoh_ * torch.log(Y + 1e-10), dim=1))
        loss += lambd * sum([torch.sum(weights ** 2) for weights in self.weights])
        return loss

    def count_params(self):
        total_params = 0
        for name, param in self.named_parameters():
            print(f"Parameter name: {name}, Shape: {param.shape}")
            total_params += param.numel()
        print(f"Total number of parameters: {total_params}")

    def plot_highest_loss_indexes(self, X, Yoh_):
        Y = self.forward(X)
        losses = torch.zeros(Yoh_.size(0))
        for i in range(Yoh_.size(0)):
            losses[i] = -torch.sum(Yoh_[i] * torch.log(Y[i] + 1e-10))

        def get_top_indexes(losses, top=10):
            loss_index_pairs = [(i, loss) for i, loss in enumerate(losses)]
            sorted_loss_index_pairs = sorted(loss_index_pairs, key=lambda x: x[1], reverse=True)
            top5_indexes = [pair[0] for pair in sorted_loss_index_pairs[:10]]
            return top5_indexes

        indexes = get_top_indexes(losses)
        fig, axes = plt.subplots(1, len(indexes), figsize=(15, 3))
        for i, idx in enumerate(indexes):
            sample = X.cpu().numpy()[idx].reshape(28, 28)
            axes[i].imshow(sample, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Sample {idx}')

        plt.show()
        return indexes


def train(model: PTDeep, X, Yoh_, param_niter=1000, param_eta=0.5, param_lambda=0):
    """Arguments:
       - X: model inputs [NxD], type: torch.Tensor
       - Yoh_: ground truth [NxC], type: torch.Tensor
       - param_niter: number of training iterations
       - param_delta: learning rate
       - param_lambda: regularization
    """
    loss_history = []

    optimizer = optim.SGD(model.parameters(), lr=param_eta)

    for i in range(param_niter):
        loss = model.get_loss(X, Yoh_, param_lambda)
        loss_history.append(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            print("iteration {}: loss {}".format(i, loss))

    return loss_history


def train_adam(model: PTDeep, X, Yoh_, param_niter=1000, param_eta=0.5, param_lambda=0):
    loss_history = []

    optimizer = optim.Adam(model.parameters(), lr=param_eta)

    for i in range(param_niter):
        loss = model.get_loss(X, Yoh_, param_lambda)
        loss_history.append(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            print("iteration {}: loss {}".format(i, loss))

    return loss_history


def train_adam_auto(model: PTDeep, X, Yoh_, param_niter=1000, param_eta=0.5, param_lambda=0, param_gamma=1-1e-4):
    loss_history = []

    optimizer = optim.Adam(model.parameters(), lr=param_eta)
    scheduler = ExponentialLR(optimizer, gamma=param_gamma)
    for i in range(param_niter):
        loss = model.get_loss(X, Yoh_, param_lambda)
        loss_history.append(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            print("iteration {}: loss {}".format(i, loss))
        scheduler.step()

    return loss_history


def train_mb(model: PTDeep, X, Yoh_, param_niter=1000, param_eta=0.5, param_lambda=0, batch_size=32):
    loss_history = []

    optimizer = optim.SGD(model.parameters(), lr=param_eta)

    N = X.shape[0]

    for i in range(param_niter):
        loss = None
        indices = torch.randperm(N)
        X_shuffled = X[indices]
        Yoh_shuffled = Yoh_[indices]

        for j in range(0, N, batch_size):
            X_batch = X_shuffled[j:j + batch_size]
            Yoh_batch = Yoh_shuffled[j:j + batch_size]

            loss = model.get_loss(X_batch, Yoh_batch, param_lambda)
            loss_history.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % 1 == 0:
            print("iteration {}: loss {}".format(i, loss))

    return loss_history


def eval(model: PTDeep, X):
    """Arguments:
       - model: type: PTLogreg
       - X: actual datapoints [NxD], type: np.array
       Returns: predicted class probabilites [NxC], type: np.array
    """
    device = next(model.parameters()).device
    print(X)
    X_tensor = torch.from_numpy(X).to(device)
    probs = model.forward(X_tensor).detach().cpu().numpy()
    classes = np.argmax(probs, axis=1)
    return classes


def PTDeep_decfun(model):
    def classify(X):
        return eval(model, X)
    return classify


if __name__ == "__main__":
    np.random.seed(100)

    #  X, Y_ = sample_gauss_2d(2, 100)
    #  X, Y_ = sample_gauss_2d(3, 100)
    X, Y_ = sample_gmm_2d(6, 2, 10)
    Yoh_ = class_to_onehot(Y_)

    # define model
    ptdeep = PTDeep([2, 10, 10, 2], "relu")
    ptdeep.count_params()

    # train
    X = torch.from_numpy(X)
    Yoh_ = torch.from_numpy(Yoh_)
    train(ptdeep, X, Yoh_, 100000, 0.03, 1e-4)

    # probabilities for train set
    Y = eval(ptdeep, X.numpy())

    accuracy, pr, M = eval_perf_multi(Y, Y_)
    print(accuracy); print(pr); print(M)

    # graph the decision surface
    bbox = (np.min(X.numpy(), axis=0), np.max(X.numpy(), axis=0))
    graph_surface(PTDeep_decfun(ptdeep), bbox, offset=0.5)
    graph_data(X, Y_, Y, special=[])

    #plt.savefig('img/PTDeep/sigmoid.png')
    plt.show()
