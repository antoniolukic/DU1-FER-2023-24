from __future__ import annotations
from sklearn.svm import SVC
import sys
sys.path.append("c:\\Users\\p51\\OneDrive\\Desktop\\Dubokoucenje1")
from lab0.data import *


class KSVMWrap:
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.clf = SVC(C=param_svm_c, gamma=param_svm_gamma)
        self.clf.fit(X, Y_)

    def predict(self, X):
        return self.clf.predict(X)

    def get_scores(self, X):
        return self.clf.decision_function(X)  # decorator

    def support(self):
        return self.clf.support_


if __name__ == "__main__":
    np.random.seed(100)

    #  X, Y_ = sample_gauss_2d(2, 100)
    X, Y_ = sample_gmm_2d(6, 2, 10)

    # define model and train
    ksvmwrap = KSVMWrap(X, Y_, param_svm_c=1, param_svm_gamma='auto')

    # probabilities for train set
    Y = ksvmwrap.predict(X)

    accuracy, pr, M = eval_perf_multi(Y, Y_)
    print(accuracy); print(pr); print(M)

    # graph the decision surface
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(ksvmwrap.get_scores, bbox, offset=0)
    graph_data(X, Y_, Y, special=[ksvmwrap.support()])

    plt.savefig('img/KSVm/4_0.png')
    plt.show()

