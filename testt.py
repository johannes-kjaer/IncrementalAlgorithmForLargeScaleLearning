from solver import *
from sgd import *
from svrg import *
from saga import *
import matplotlib.pyplot as plt
from mnist import MnistDataloader


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

    solver = LinearRegressionSolver(X, Y, my_SAGA)
    solver.solve()
    stats: Statistics = solver.get_stats()
    print(stats)
    #print("Real w: ", w)
    #print("Approximated w: ", solver.get_sol())
    plt.yscale("log")
    stats.plot_gradient_norm()
    plt.show()


def isolate_number(Y, number):
    return np.array([1 if y == number else -1 for y in Y])


def mnist_test():
    (X_train, Y_train), (X_test, Y_test) = MnistDataloader().load_data(stack_images=True)
    number = 5
    Y_train_isolate = isolate_number(Y_train, number)
    Y_test_isolate = isolate_number(Y_test, number)

    n = X_train.shape[0]
    eta = 0.0000001
    max_epochs = 10
    precision = 10**-5
    l = 10**-2
    my_SGD = lambda f, dim: SGD(f, dim, eta=eta, max_epochs=max_epochs, precision=precision)
    my_SVRG = lambda f, dim: SVRG(f, dim, m=2*n, eta=eta, max_epochs=max_epochs, precision=precision)
    my_SAGA = lambda f, dim: SAGA(f, dim, eta=eta, max_epochs=max_epochs, precision=precision)

    solver = LogisticRegressionSolver(X_train, Y_train_isolate, my_SVRG, l=l)
    solver.solve()
    stats: Statistics = solver.get_stats()
    print(stats)
    print("Training error :", solver.training_error())
    print("Testing error :", solver.error(X_test, Y_test_isolate))
    plt.yscale("log")
    stats.plot_gradient_norm()
    plt.show()


if __name__ == "__main__":
    mnist_test()
