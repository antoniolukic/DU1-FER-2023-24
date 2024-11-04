from __future__ import annotations
import torch
import torch.nn as nn
from torch import optim as optim
import sys
sys.path.append("c:\\Users\\p51\\OneDrive\\Desktop\\Dubokoucenje1")
from lab0.data import *


class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
        """
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(size=(D, C), dtype=torch.float64), requires_grad=True)
        self.b = torch.nn.Parameter(torch.zeros(C, dtype=torch.float64, requires_grad=True))

    def forward(self, X):
        scores = torch.mm(X, self.W) + self.b
        probs = torch.softmax(scores, dim=1)
        return probs

    def get_loss(self, X, Yoh_, lambd):
        Y = self.forward(X)
        loss = -torch.mean(torch.sum(Yoh_ * torch.log(Y + 1e-10), dim=1))
        loss += lambd * torch.sum(self.W ** 2)
        return loss


def train(model: PTLogreg, X, Yoh_, param_niter=1000, param_eta=0.5, param_lambda=0):
    """Arguments:
       - X: model inputs [NxD], type: torch.Tensor
       - Yoh_: ground truth [NxC], type: torch.Tensor
       - param_niter: number of training iterations
       - param_delta: learning rate
       - param_lambda: regularization
    """

    # inicijalizacija optimizatora
    optimizer = optim.SGD(model.parameters(), lr=param_eta)

    # petlja učenja
    for i in range(param_niter):
        loss = model.get_loss(X, Yoh_, param_lambda)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            print("iteration {}: loss {}".format(i, loss))


def eval(model: PTLogreg, X):
    """Arguments:
       - model: type: PTLogreg
       - X: actual datapoints [NxD], type: np.array
       Returns: predicted class probabilites [NxC], type: np.array
    """
    # ulaz je potrebno pretvoriti u torch.Tensor
    # izlaze je potrebno pretvoriti u numpy.array
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()

    X_tensor = torch.from_numpy(X)
    probs = model.forward(X_tensor).detach().cpu().numpy()
    classes = np.argmax(probs, axis=1)
    return classes


def PTLogreg_decfun(model):
    def classify(X):
        return eval(model, X)
    return classify


if __name__ == "__main__":
    np.random.seed(100)

    X, Y_ = sample_gauss_2d(2, 100)
    #  X, Y_ = sample_gauss_2d(3, 100)
    Yoh_ = class_to_onehot(Y_)

    # define model
    ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])

    # train
    X = torch.from_numpy(X)
    Yoh_ = torch.from_numpy(Yoh_)
    train(ptlr, X, Yoh_, 1000, 0.01, 0)

    # probabilities for train set
    Y = eval(ptlr, X.numpy())

    accuracy, pr, M = eval_perf_multi(Y, Y_)
    print(accuracy); print(pr); print(M)

    # graph the decision surface
    bbox = (np.min(X.numpy(), axis=0), np.max(X.numpy(), axis=0))
    graph_surface(PTLogreg_decfun(ptlr), bbox, offset=0.5)
    graph_data(X, Y_, Y, special=[])

    #  plt.savefig('img/iter_40000_eta_01_reg_0_c_3.png')
    plt.show()
