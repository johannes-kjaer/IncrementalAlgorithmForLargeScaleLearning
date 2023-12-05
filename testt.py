from solver import *
from sgd import *
from svrg import *
from saga import *
from sdca import *
import matplotlib.pyplot as plt
from mnist import MnistDataloader
from data_generation import *
import sys


def main():
    m = 100      # dimension of the values
    n = 1000      # size of the data set
    X = np.array([[random.random() for _ in range(m)] for _ in range(n)], dtype=DTYPE)
    w = np.array([random.random() for _ in range(m)], dtype=DTYPE)
    Y = X @ w

    eta = 0.005
    max_epochs = 100
    precision = 10**-5
    my_SGD = lambda f, dim: SGD(f, dim, eta=eta, max_epochs=max_epochs, precision=precision)
    my_SVRG = lambda f, dim: SVRG(f, dim, m=2*n, eta=eta, max_epochs=max_epochs, precision=precision)
    my_SAGA = lambda f, dim: SAGA(f, dim, eta=eta, max_epochs=max_epochs, precision=precision)
    my_SDCA = lambda: SDCA(X=X, Y=Y, C=C, max_epochs=max_epochs, precision=precision)

    solver = LinearRegressionSolver(X, Y, my_SAGA)
    solver.solve()
    stats: Statistics = solver.get_stats()
    print(stats)
    #print("Real w: ", w)
    #print("Approximated w: ", solver.get_w())
    plt.yscale("log")
    stats.plot_gradient_norm()
    plt.show()


def probability(x: np.ndarray, y: int, w: np.ndarray):
    return 1 / (1 + np.exp(-y * w @ x))

def predicted_label(x: np.ndarray, w: np.ndarray):
    return 1 if probability(x, 1, w) > 0.5 else -1

def logistic_regression_test():
    m = 100      # dimension of the values
    n = 1000      # size of the data set
    X, cov, mean = generateRandomData(m, n)
    X, Y, cov, mean = generateCompleteData(m, n, ratio=0)

    max_epochs = 100
    precision = 10**-5
    C = 0.0001
    eta = 0.00001
    l = 1/(2*C)
    l = 0.0
    
    my_SGD = lambda f, dim: SGD(f, dim, eta=eta, max_epochs=max_epochs, precision=precision)
    my_SVRG = lambda f, dim: SVRG(f, dim, m=2*n, eta=eta, max_epochs=max_epochs, precision=precision)
    my_SAGA = lambda f, dim: SAGA(f, dim, eta=eta, max_epochs=max_epochs, precision=precision)
    my_SDCA = lambda: SDCA(X=X, Y=Y, C=C, max_epochs=max_epochs, precision=precision, keep_Q=True)

    methods = {"sgd": my_SGD, "svrg": my_SVRG, "saga": my_SAGA, "sdca": my_SDCA}
    method = "svrg" if len(sys.argv) < 2 else sys.argv[1]
    if method == "sdca":
        solver = SDCASolver(X, Y, my_SDCA)
    else:
        solver = LogisticRegressionSolver(X, Y, methods[method], l=l)
    solver.solve()
    stats: Statistics = solver.get_stats()
    print(stats)
    print("Training error :", solver.training_error())
    # print("Real w: ", w)
    # print("Approximated w: ", solver.get_w())
    plt.yscale("log")
    stats.plot_gradient_norm()
    plt.show()


def isolate_number(Y, number):
    return np.array([1 if y >= number else -1 for y in Y])


def isolate_numbers(X, Y, n1, n2):
    X_ = []
    Y_ = []
    for x, y in zip(X, Y):
        if y == n1:
            X_.append(x)
            Y_.append(-1)
        elif y == n2:
            X_.append(x)
            Y_.append(1)
    return np.array(X_), np.array(Y_)


def mnist_test():
    (X_train, Y_train), (X_test, Y_test) = MnistDataloader().load_data(stack_images=True)
    number = 5
    n1, n2 = 2, 6
    X_train, Y_train = isolate_numbers(X_train, Y_train, n1, n2)
    X_test, Y_test = isolate_numbers(X_test, Y_test, n1, n2)

    n = X_train.shape[0]
    eta = 0.000001
    max_epochs = 10
    precision = 10**-5
    l = 10**-2
    C = 0.0000001
    l = 1/(2*C)
    my_SGD = lambda f, dim: SGD(f, dim, eta=eta, max_epochs=max_epochs, precision=precision)
    my_SVRG = lambda f, dim: SVRG(f, dim, m=2*n, eta=eta, max_epochs=max_epochs, precision=precision)
    my_SAGA = lambda f, dim: SAGA(f, dim, eta=eta, max_epochs=max_epochs, precision=precision)
    my_SDCA = lambda: SDCA(X=X_train, Y=Y_train, C=C, max_epochs=max_epochs, precision=precision)

    solver = LogisticRegressionSolver(X_train, Y_train, my_SVRG, l=l)
    #solver = SDCASolver(X_train, Y_train, my_SDCA)
    solver.solve()
    stats: Statistics = solver.get_stats()
    #print(solver.get_w())
    print(stats)
    print("Training error :", solver.training_error())
    print("Testing error :", solver.error(X_test, Y_test))
    plt.yscale("log")
    stats.plot_gradient_norm()
    plt.show()


if __name__ == "__main__":
    logistic_regression_test()
