from __future__ import annotations
from data import *
import numpy as np


def logreg_train(X: np.ndarray, Y_: np.ndarray, param_eta=0.001, param_niter=4000, param_lambda=0):
    """" Argumenti
          X:    podatci, np.array NxD
          Y_:   prave oznake primjera

      Povratne vrijednosti
          W, b: parametri viserazredne logisticke regresije"""

    N = len(X)
    D = len(X[0])
    C = max(Y_) + 1

    W = np.random.normal(0, 1, (D, C))
    b = np.zeros(C)
    Y_onehot = class_to_onehot(Y_)

    # gradient descent
    for i in range(param_niter + 1):
        # classification scores
        scores = X @ W + b  # N x C
        exp_scores = np.exp(scores)  # N x C

        # denominator
        sum_exp = np.sum(exp_scores, axis=1, keepdims=True)  # N x 1

        # probabilities of class c_1
        probs = exp_scores / sum_exp  # N x 1
        log_probs = np.log(probs + 1e-10)

        # loss
        loss = -np.sum(log_probs[range(N), Y_]) / N  # scalar

        if i % 100 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivations of loss by classification scores
        dL_dscores = probs - Y_onehot

        # gradient of parameters
        grad_W = (dL_dscores.T @ X).T / N + 2 * param_lambda * W  # C x D (ili D x C)
        grad_b = np.sum(dL_dscores, axis=0) / N  # C x 1 (ili 1 x C)

        # update parameters
        W += - param_eta * grad_W
        b += - param_eta * grad_b

    return W, b


def logreg_classify(X: np.ndarray, W: np.ndarray, b: np.ndarray):
    """Argumenti
          X:    podatci, np.array (N*C)xD
          W, b: parametri logističke regresije

      Povratne vrijednosti
          probs: vjerojatnosti razreda c1"""

    scores = X @ W + b
    exp_logits = np.exp(scores)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probs


def logreg_decfun(W, b):
    def classify(X):
        return np.argmax(logreg_classify(X, W, b), axis=1)
    return classify


if __name__ == "__main__":
    np.random.seed(100)

    X, Y_ = sample_gauss_2d(3, 100)
    W, b = logreg_train(X, Y_, param_eta=0.01, param_niter=40000)

    # evaluate the model on the training dataset
    probs = logreg_classify(X, W, b)
    Y = np.argmax(probs, axis=1)

    accuracy, pr, M = eval_perf_multi(Y, Y_)
    print(accuracy); print(pr); print(M)

    # graph the decision surface
    dec_fun = logreg_decfun(W, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(dec_fun, bbox, offset=0.5)
    graph_data(X, Y_, Y, special=[])

    plt.show()
