from __future__ import annotations
from data import *
import numpy as np


def binlogreg_train(X: np.ndarray, Y_: np.ndarray, param_eta=0.001, param_niter=4000, param_lambda=0):
    """Argumenti
      X:  podatci, np.array NxD
      Y_: indeksi razreda, np.array Nx1

      Povratne vrijednosti
      w, b: parametri logističke regresije"""

    N = len(X)
    D = len(X[0])
    w = np.random.normal(0, 1, D)
    b = 0

    # gradient descent
    for i in range(param_niter + 1):
        # classification scores
        scores = X @ w + b  # N x 1

        # probabilities of class c_1
        probs = 1 / (1 + np.exp(-scores))  # N x 1

        # loss
        loss = -np.mean(Y_ * np.log(probs + 1e-10) + (1 - Y_) * np.log(1 - probs + 1e-10))  # scalar
        loss += param_lambda / 2 * np.sum(w ** 2)

        if i % 100 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivations of loss by classification scores
        dL_dscores = probs - Y_  # N x 1

        # gradient of parameters
        grad_w = X.T @ dL_dscores + param_lambda * w  # D x 1
        grad_b = np.sum(dL_dscores) / N  # 1 x 1

        # update parameters
        w += - param_eta * grad_w
        b += - param_eta * grad_b

    return w, b


def binlogreg_classify(X: np.ndarray, w: np.ndarray, b: float):
    """Argumenti
          X:    podatci, np.array NxD
          w, b: parametri logističke regresije

      Povratne vrijednosti
          probs: vjerojatnosti razreda c1"""

    s = X @ w + b
    return 1 / (1 + np.exp(-s))


def binlogreg_decfun(w, b):
    def classify(X):
        return binlogreg_classify(X, w, b)
    return classify


if __name__ == "__main__":
    np.random.seed(100)

    X, Y_ = sample_gauss_2d(2, 100)
    w, b = binlogreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w, b)
    Y = np.round(probs)

    accuracy, recall, precision = eval_perf_binary(Y, Y_)
    AP = eval_AP(Y_[probs.argsort()])
    print(accuracy, recall, precision, AP)

    # graph the decision surface
    dec_fun = binlogreg_decfun(w, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(dec_fun, bbox, offset=0.5)
    graph_data(X, Y_, Y, special=[])

    plt.show()
