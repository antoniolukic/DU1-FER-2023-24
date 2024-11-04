from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


class Random2DGaussian:
    minx = 0
    maxx = 10
    miny = 0
    maxy = 10
    scale_cov = 5

    def __init__(self):
        rx, ry = self.maxx - self.minx, self.maxy - self.miny
        self.mean = (self.minx, self.miny)
        self.mean += np.random.random_sample(2) * (rx, ry)

        eig_vals = np.random.random_sample(2)
        eig_vals *= (rx / self.scale_cov, ry / self.scale_cov)
        eig_vals **= 2

        theta = np.random.random_sample() * np.pi * 2
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        self.cov = R.T @ np.diag(eig_vals) @ R

    def get_sample(self, n):
        return np.random.multivariate_normal(self.mean, self.cov, n)


def sample_gauss_2d(C, N):
    # create the distributions and ground truth labels
    Gs = []
    Ys = []
    for i in range(C):
        Gs.append(Random2DGaussian())
        Ys.append(i)

    # sample the dataset
    X = np.vstack([G.get_sample(N) for G in Gs])
    Y_ = np.hstack([[Y] * N for Y in Ys])

    return X, Y_


def sample_gmm_2d(K, C, N):  # n_components, n_classes, n_samples
    # create the distributions and ground truth labels
    """
      X  ... podatci u matrici [K·N x 2]
      Y_ ... indeksi razreda podataka [1 x K·N]
    """
    Gs = []
    Ys = []
    for i in range(K):
        Gs.append(Random2DGaussian())
        Ys.append(np.random.randint(C))

    # sample the dataset
    X = np.vstack([G.get_sample(N) for G in Gs])
    Y_ = np.hstack([[Y] * N for Y in Ys])

    return X, Y_


def eval_perf_binary(Y, Y_):
    tp = sum(np.logical_and(Y == Y_, Y_ == True))
    tn = sum(np.logical_and(Y == Y_, Y_ == False))
    fp = sum(np.logical_and(Y == True, Y_ == False))
    fn = sum(np.logical_and(Y == False, Y_ == True))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy, recall, precision


def eval_perf_multi(Y, Y_):
    pr = []
    n = max(Y_) + 1
    M = np.bincount(n * Y_ + Y, minlength=n * n).reshape(n, n)
    for i in range(n):
        tp_i = M[i, i]
        fn_i = np.sum(M[i, :]) - tp_i
        fp_i = np.sum(M[:, i]) - tp_i
        tn_i = np.sum(M) - fp_i - fn_i - tp_i
        recall_i = 0 if tp_i + fn_i <= 0 else tp_i / (tp_i + fn_i)
        precision_i = 0 if tp_i + fp_i <= 0 else tp_i / (tp_i + fp_i)
        pr.append((recall_i, precision_i))

    accuracy = np.trace(M) / np.sum(M)

    return accuracy, pr, M


def eval_AP(ranked_labels):
    n = len(ranked_labels)
    pos = sum(ranked_labels)
    neg = n - pos
    tp, tn, fn, fp = pos, 0, 0, neg
    sumprec = 0

    for x in ranked_labels:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if x:
            sumprec += precision

        tp -= x
        fn += x
        fp -= not x
        tn += not x

    return sumprec / pos


def my_dummy_decision(X):
    scores = X[:, 0] + X[:, 1] - 5
    return scores


def class_to_onehot(Y):
    Y_onehot = np.zeros((len(Y), max(Y) + 1))
    Y_onehot[range(len(Y)), Y] = 1
    return Y_onehot


def graph_data(X, Y_, Y, special=[]):
    palette = ([0.5, 0.5, 0.5], [1, 1, 1], [0.2, 0.2, 0.2])
    colors = np.tile([0.0, 0.0, 0.0], (Y_.shape[0], 1))
    for i in range(len(palette)):
        colors[Y_ == i] = palette[i]

    # sizes of the datapoint markers
    sizes = np.repeat(20, len(Y_))
    sizes[special] = 40

    # draw the correctly classified datapoints
    good = (Y_ == Y)
    plt.scatter(X[good, 0], X[good, 1], c=colors[good],
                s=sizes[good], marker='o', edgecolors='black')

    # draw the incorrectly classified datapoints
    bad = (Y_ != Y)
    plt.scatter(X[bad, 0], X[bad, 1], c=colors[bad],
                s=sizes[bad], marker='s', edgecolors='black')


def graph_surface(function, rect, offset=0.5, width=256, height=256):
    """ Creates a surface plot (visualize with plt.show)

    Arguments:
      function: surface to be plotted
      rect:     function domain provided as:
                ([x_min,y_min], [x_max,y_max])
      offset:   the level plotted as a contour plot
      width: width in pixels
      height: height in pixels

    Returns:
      None """

    lsw = np.linspace(rect[0][1], rect[1][1], width)
    lsh = np.linspace(rect[0][0], rect[1][0], height)
    xx0, xx1 = np.meshgrid(lsh, lsw)
    grid = np.stack((xx0.flatten(), xx1.flatten()), axis=1)

    # get the values and reshape them
    values = function(grid).reshape((width, height))

    # fix the range and offset
    delta = offset if offset else 0
    maxval = max(np.max(values) - delta, - (np.min(values) - delta))

    # draw the surface and the offset
    plt.pcolormesh(xx0, xx1, values, vmin=delta - maxval, vmax=delta + maxval, cmap='jet')
    plt.contour(xx0, xx1, values, colors='black', levels=[offset])


if __name__ == "__main__":
    np.random.seed(100)

    X, Y_ = sample_gmm_2d(4, 2, 30)
    #  X, Y_ = sample_gauss_2d(2, 100)

    # get the class predictions
    Y = my_dummy_decision(X) > 0.5

    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(my_dummy_decision, bbox, offset=0.5)
    graph_data(X, Y_, Y)

    plt.show()
