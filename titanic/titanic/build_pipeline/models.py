from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier


def define_model():
    return LogisticRegression(max_iter=200)

def define_svm_model():
    return svm.SVC()

def define_nn_model():
    return MLPClassifier()
