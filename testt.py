from solver import *
from sgd import *
from gd import *
from svrg import *
from saga import *
from sdca import *
import matplotlib.pyplot as plt
from mnist import MnistDataloader
from data_generation import *
from newDataGeneration import multinormalRandomData
from cross_validation import *


def get_logistic_regression_solver(X_train: np.ndarray, Y_train: np.ndarray, eta: float, l: float, C: float, max_epochs: int, precision: float, keep_gradient: bool) -> tuple[str, Solver]:
    n = X_train.shape[0]
    my_GD = lambda f, dim: GD(f, dim, eta=eta, max_epochs=max_epochs, precision=precision)
    my_SGD = lambda f, dim: SGD(f, dim, eta=eta, max_epochs=max_epochs, precision=precision, keep_gradient=keep_gradient)
    my_SVRG = lambda f, dim: SVRG(f, dim, m=2*n, eta=eta, max_epochs=max_epochs, precision=precision)
    my_SAGA = lambda f, dim: SAGA(f, dim, eta=eta, max_epochs=max_epochs, precision=precision, keep_gradient=keep_gradient)
    my_SDCA = lambda: SDCA(X=X_train, Y=Y_train, C=C, max_epochs=max_epochs, precision=precision, keep_Q=False, keep_gradient=keep_gradient)

    methods = {"gd": my_GD, "sgd": my_SGD, "svrg": my_SVRG, "saga": my_SAGA, "sdca": my_SDCA}
    method = get_method_from_user_input()
    if method == "sdca":
        solver = SDCASolver(X_train, Y_train, my_SDCA)
    else:
        solver = LogisticRegressionSolver(X_train, Y_train, methods[method], l=l)
    return method, solver


def logistic_regression_test():
    m = 100      # dimension of the values
    n = 1000      # size of the data set
    c1 = 0.1

    X_train, Y_train, X_test, Y_test, w, additional = generateCompleteData(
        dim = m,
        nb_samples = n,
        sample_generator = multinormalRandomData,
        label_generator = lambda X, w: generateLabels(X, w, sigma=0.1),
        c1 = c1,
        c2 = 1.0,
    )

    K = 5  # for cross validation
    max_epochs = 20
    precision = 10**-5
    C = 0.01
    eta = 0.1
    #l = 1/(2*C*X_train.shape[0])
    l = 0.1
    svrg_m = 2*n
    keep_gradient = True

    # # cross validation for the learning rate eta
    # solver = complete_cross_validation(X_train, Y_train, K, max_epochs, precision, keep_gradient, 
    #                                    param_values=[10.0, 1.0, 0.1, 0.01, 0.001, 0.0001], eta=None, l=l, C=C, svrg_m=svrg_m)

    solver = cross_validate_l(X_train, Y_train, K, max_epochs, precision, keep_gradient, 
                                       param_values=[100.0, 10.0, 1.0, 0.1, 0.01, 0.001], eta=eta, C=C, svrg_m=svrg_m)
    solver.solve()

    stats: Statistics = solver.get_stats()
    print(stats)
    print("Training error :", solver.training_error())
    print("Testing error :", solver.error(X_test, Y_test))

    print("\nReal w training error :", solver.training_error(w=w))
    print("Real w testing error esting error :", solver.error(X_test, Y_test, w=w))

    # print("Real w: ", w)
    # print("Approximated w: ", solver.get_w())
    plt.yscale("log")
    stats.plot_objective_function()
    plt.title(f"{get_method_from_user_input()} with c1={c1}")
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
    n1, n2 = 2, 6
    X_train, Y_train = isolate_numbers(X_train, Y_train, n1, n2)
    X_test, Y_test = isolate_numbers(X_test, Y_test, n1, n2)

    n = X_train.shape[0]
    eta = 0.00008
    max_epochs = 20
    precision = 10**-5
    l = 1.0
    C = 1/(2*l*n)
    
    method, solver = get_logistic_regression_solver(X_train, Y_train, eta, l, C, max_epochs, precision, keep_gradient=True)
    solver.solve()
    stats: Statistics = solver.get_stats()
    #print(solver.get_w())
    print(stats)
    print("Training error :", solver.training_error())
    print("Testing error :", solver.error(X_test, Y_test))
    #plt.yscale("log")
    #stats.plot_gradient_norm()
    stats.plot_objective_function()
    plt.title(f"{method} MNIST")
    plt.show()


if __name__ == "__main__":
    logistic_regression_test()
