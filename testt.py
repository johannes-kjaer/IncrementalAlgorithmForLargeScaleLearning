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


def get_logistic_regression_solver(X_train: np.ndarray, Y_train: np.ndarray, eta: float, l: float, C: float, svrg_m_coef: float, max_epochs: int, precision: float, keep_gradient: bool, method: str=None) -> tuple[str, Solver]:
    my_GD = lambda f, dim: GD(f, dim, eta=eta, max_epochs=max_epochs, precision=precision)
    my_SGD = lambda f, dim: SGD(f, dim, eta=eta, max_epochs=max_epochs, precision=precision, keep_gradient=keep_gradient)
    my_SVRG = lambda f, dim: SVRG(f, dim, m_coef=svrg_m_coef, eta=eta, max_epochs=max_epochs, precision=precision)
    my_SAGA = lambda f, dim: SAGA(f, dim, eta=eta, max_epochs=max_epochs, precision=precision, keep_gradient=keep_gradient)
    my_SDCA = lambda: SDCA(X=X_train, Y=Y_train, C=C, max_epochs=max_epochs, precision=precision, keep_Q=False, keep_gradient=keep_gradient)

    methods = {"gd": my_GD, "sgd": my_SGD, "svrg": my_SVRG, "saga": my_SAGA, "sdca": my_SDCA}
    if method is None:
        method = get_method_from_user_input()
    if method == "sdca":
        solver = SDCASolver(X_train, Y_train, my_SDCA)
    else:
        solver = LogisticRegressionSolver(X_train, Y_train, methods[method], l=l)
    return method, solver


def logistic_regression_test():
    m = 1000      # dimension of the values
    n = 10000      # size of the data set
    c1 = 0.1

    X_train, Y_train, X_test, Y_test, w, additional = generateCompleteData(
        dim = m,
        nb_samples = n,
        sample_generator = multinormalRandomData,
        label_generator = lambda X, w: generateLabels(X, w, sigma=0.0),
        c1 = c1,
        c2 = 1.0,
    )

    K = 5  # for cross validation
    max_epochs = 20
    precision = 10**-5
    C = 0.01
    eta = 0.1
    #l = 1 / (2*C*X_train.shape[0])
    l = 0.1
    C = 1 / (2*l*X_train.shape[0])
    svrg_m_coef = 2.0
    keep_gradient = True

    # cross validation for the learning rate eta
    solver = complete_cross_validation(X_train, Y_train, K, max_epochs, precision, keep_gradient, 
                                       param_values=[10**e for e in float_range(-1.5, -3.5, -0.5)], eta=None, l=l, C=C, svrg_m_coef=svrg_m_coef)
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


def logistic_regression_cmp():
    # cross validated eta for m=100, n=1000, c1=0.1, sigma=0 (noise on Y)
    # gd: 1.0
    # sgd: 0.1
    # svrg: 10**-2.5
    # saga: 0.001
    # sdca: _

    # cross validated eta for m=1000, n=10000, c1=0.1, sigma=0
    # gd: 1
    # sgd: 0.01
    # svrg: 10**-3.5
    # saga: 0.001

    m = 1000      # dimension of the values
    n = 10000      # size of the data set
    c1 = 0.1

    X_train, Y_train, X_test, Y_test, w, additional = generateCompleteData(
        dim = m,
        nb_samples = n,
        sample_generator = multinormalRandomData,
        label_generator = lambda X, w: generateLabels(X, w, sigma=0.1),
        c1 = c1,
        c2 = 1.0,
    )

    max_epochs = 5
    precision = 10**-8
    C = 0.01
    #eta = 0.1
    #l = 1 / (2*C*X_train.shape[0])
    l = 0.1
    C = 1 / (2*l*X_train.shape[0])
    svrg_m_coef = 2.0
    keep_gradient = True

    optimal_eta = {"gd": 1.0, "sgd": 0.01, "svrg": 10**-3.5, "saga": 10**-3}#, "sdca": 1.0}    # not used for sdca
    for method, eta in optimal_eta.items():
        _, solver = get_logistic_regression_solver(X_train, Y_train, method=method, eta=eta, l=l, C=C, svrg_m_coef=svrg_m_coef, max_epochs=max_epochs, precision=precision, keep_gradient=keep_gradient)
        solver.solve()
        stats = solver.get_stats()
        stats.plot_objective_function(label=method)
        print(stats)
        print("Training error :", solver.training_error())
        print("Testing error :", solver.error(X_test, Y_test))
        print()
    print("\nReal w training error :", solver.training_error(w=w))
    print("Real w testing error esting error :", solver.error(X_test, Y_test, w=w))
    plt.legend()
    plt.yscale("log")
    plt.show()


def cmp_methods_logistic_regression(X_train, Y_train, X_test, Y_test, optimal_etas, l, C, svrg_m_coef, max_epochs, precision, keep_gradient, real_w=None):
    for method, eta in (optimal_etas.items() if isinstance(optimal_etas, dict) else optimal_etas):
        _, solver = get_logistic_regression_solver(X_train, Y_train, method=method, eta=eta, l=l, C=C, svrg_m_coef=svrg_m_coef, max_epochs=max_epochs, precision=precision, keep_gradient=keep_gradient)
        solver.solve()
        stats = solver.get_stats()
        stats.plot_objective_function(label=method)
        print(stats)
        print("Training error :", solver.training_error())
        print("Testing error :", solver.error(X_test, Y_test))
        print()

    if real_w is not None:
        print("\nReal w training error :", solver.training_error(w=real_w))
        print("Real w testing error esting error :", solver.error(X_test, Y_test, w=real_w))
    plt.legend()
    plt.yscale("log")
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


def get_mnist_data():
    (X_train, Y_train), (X_test, Y_test) = MnistDataloader().load_data(stack_images=True)
    n1, n2 = 2, 6
    X_train, Y_train = isolate_numbers(X_train, Y_train, n1, n2)
    X_test, Y_test = isolate_numbers(X_test, Y_test, n1, n2)
    return X_train, Y_train, X_test, Y_test


def mnist_test():
    # cross validated svrg_m_coef: 1.9

    # cross validated eta
    # gd: 10**-6
    # sgd: 10**-4.5
    # sgd constant learning rate: 10**-6
    # svrg: 10**-6.5
    # saga: 10**-6.5

    # cross validated l
    # gd: 0.1   (actually seems to perform exactly the same as long as l is not bigger than 1000)
    # sgd: 10**-0.5
    # svrg: 

    X_train, Y_train, X_test, Y_test = get_mnist_data()

    n = X_train.shape[0]
    K = 5   # for cross validation
    #eta = 0.00008
    max_epochs = 10
    precision = 10**-5
    l = 1.0
    C = 1/(2*l*n)
    svrg_m_coef = 1.9

    #method, solver = get_logistic_regression_solver(X_train, Y_train, eta, l, C, svrg_m_coef, max_epochs, precision, keep_gradient=True)
    optimal_etas = {"gd": 10**-5.5, "sgd": 10**-4.5, "svrg": 10**-6.5, "saga": 10**-6.5}
    eta = optimal_etas[get_method_from_user_input()]
    solver = complete_cross_validation(X_train, Y_train, K, max_epochs, precision, keep_gradient=True, 
                                       param_values=np.power(10, [-np.inf, -1.0, -0.5, 0.0, 0.5]), eta=eta, l=None, C=C, svrg_m_coef=svrg_m_coef)
    solver.solve()
    stats: Statistics = solver.get_stats()
    #print(solver.get_w())
    print(stats)
    print("Training error :", solver.training_error())
    print("Testing error :", solver.error(X_test, Y_test))
    #plt.yscale("log")
    #stats.plot_gradient_norm()
    stats.plot_objective_function()
    plt.yscale("log")
    plt.title(f"{get_method_from_user_input()} MNIST")
    plt.show()


def mnist_cmp():
    X_train, Y_train, X_test, Y_test = get_mnist_data()
    n = X_train.shape[0]
    max_epochs = 40
    precision = 10**-5
    l = 1.0
    C = 1/(2*l*n)
    svrg_m_coef = 2.0

    optimal_etas = {"gd": 10**-5.5, "sgd": 10**-4.5, "svrg": 10**-6.5, "saga": 10**-6.5}
    optimal_etas = [("gd", 10**-5), ("gd", 10**-5.5), ("sgd", 10**-4.5), ("svrg", 10**-6.5), ("saga", 10**-6.5)]
    cmp_methods_logistic_regression(X_train, Y_train, X_test, Y_test, optimal_etas, l, C, svrg_m_coef, max_epochs, precision, keep_gradient=True)


if __name__ == "__main__":
    mnist_test()
