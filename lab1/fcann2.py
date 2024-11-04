from __future__ import annotations
import sys
sys.path.append("c:\\Users\\p51\\OneDrive\\Desktop\\Dubokoucenje1")
from lab0.data import *
from lab0.utils import *
import numpy as np


def fcann2_forward_pass(X, W1, b1, W2, b2):
    s1 = X @ W1 + b1  # N x H
    h1 = relu(s1)
    s2 = h1 @ W2 + b2  # N x C
    P = softmax(s2)  # N x C
    return s1, h1, s2, P


def fcann2_loss(Y_onehot, P, W1, W2, lambd):
    N = len(Y_onehot)
    loss = -np.sum(Y_onehot * np.log(P + 1e-10)) / N
    loss += lambd * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
    return loss


def fcann2_train(X: np.ndarray, Y_: np.ndarray, param_eta=0.001, param_niter=4000, param_lambda=0, param_hidden=5):
    """" Argumenti
          X:    podatci, np.array NxD
          Y_:   prave oznake primjera

      Povratne vrijednosti
          W1, W2, b1, b2: parametri unaprijedne mreze"""

    N = len(X)
    D = len(X[0])
    C = max(Y_) + 1
    H = param_hidden

    W1, W2 = np.random.randn(D, H), np.random.randn(H, C)
    b1, b2 = np.zeros((1, H)), np.zeros((1, C))
    Y_onehot = class_to_onehot(Y_)

    # gradient descent
    for i in range(param_niter + 1):
        # classification scores
        s1, h1, s2, P = fcann2_forward_pass(X, W1, b1, W2, b2)
        loss = fcann2_loss(Y_onehot, P, W1, W2, param_lambda)

        Gs2 = (P - Y_onehot) / N
        grad_W2 = (Gs2.T @ h1).T + 2 * param_lambda * W2
        grad_b2 = np.sum(Gs2, axis=0, keepdims=True)
        Gh1 = Gs2 @ W2.T
        Gs1 = Gh1 * (s1 > 0)
        grad_W1 = (Gs1.T @ X).T + 2 * param_lambda * W1
        grad_b1 = np.sum(Gs1, axis=0, keepdims=True)

        if i % 100 == 0:
            print("iteration {}: loss {}".format(i, loss))

        W1 -= param_eta * grad_W1
        b1 -= param_eta * grad_b1
        W2 -= param_eta * grad_W2
        b2 -= param_eta * grad_b2

    return W1, b1, W2, b2


def fcann2_classify(X: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2):
    """Argumenti
          X:    podatci, np.array (N*C)xD
          W1, b1, W2, b2: parametri unaprijedne mreze

      Povratne vrijednosti
          probs: vjerojatnosti razreda"""

    s1, h1, s2, P = fcann2_forward_pass(X, W1, b1, W2, b2)
    return P


def fcann2_decfun(W1, b1, W2, b2):
    def classify(X):
        return np.argmax(fcann2_classify(X, W1, b1, W2, b2), axis=1)
    return classify


if __name__ == "__main__":
    np.random.seed(100)

    X, Y_ = sample_gmm_2d(6, 2, 10)
    W1, b1, W2, b2 = fcann2_train(X, Y_, param_eta=0.05, param_niter=100000, param_lambda=1e-3, param_hidden=5)

    # evaluate the model on the training dataset
    probs = fcann2_classify(X, W1, b1, W2, b2)
    Y = np.argmax(probs, axis=1)

    accuracy, pr, M = eval_perf_multi(Y, Y_)
    print(accuracy); print(pr); print(M)

    # graph the decision surface
    dec_fun = fcann2_decfun(W1, b1, W2, b2)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(dec_fun, bbox, offset=0.5)
    graph_data(X, Y_, Y, special=[])

    plt.show()
