from solver import *
from sgd import *
from svrg import *
from saga import *
from sdca import *
import matplotlib.pyplot as plt
from mnist import MnistDataloader
#from data_generation import *
from newDataGeneration import *
import sys


def compare(X_train, Y_train, X_test, Y_test, max_epochs, precision, eta, C, l, repetitions=5):
    def compute(keep_gradient, reps):
        my_SGD = lambda f, dim: SGD(f, dim, eta=eta, max_epochs=max_epochs, precision=precision, keep_gradient=keep_gradient)
        my_SVRG = lambda f, dim: SVRG(f, dim, m=2*n, eta=eta, max_epochs=max_epochs, precision=precision)
        my_SAGA = lambda f, dim: SAGA(f, dim, eta=eta, max_epochs=max_epochs, precision=precision, keep_gradient=keep_gradient)
        my_SDCA = lambda: SDCA(X=X_train, Y=Y_train, C=C, max_epochs=max_epochs, precision=precision, keep_Q=False, keep_gradient=keep_gradient)

        results = {"sgd": [], "svrg": [], "saga": [], "sdca": []}
        methods = {"sgd": my_SGD, "svrg": my_SVRG, "saga": my_SAGA, "sdca": my_SDCA}
        for method in methods:
            for i in range(reps):
                print(f"Starting iteration {i} of {method}")
                if method == "sdca":
                    solver = SDCASolver(X_train, Y_train, my_SDCA)
                else:
                    solver = LogisticRegressionSolver(X_train, Y_train, methods[method], l=l)
                solver.solve()
                stats: Statistics = solver.get_stats()
                stats.training_error = solver.training_error()
                stats.testing_error = solver.error(X_test, Y_test)
                results[method].append(stats)
        return results

    n = X_train.shape[0]
    results_no_gradient = compute(False, repetitions)
    # results_gradient = compute(True, 1)
    methods = ["sgd", "svrg", "saga", "sdca"]
    times =  {}
    training_errors = {}
    testing_errors = {}
    for method in methods:
        res = results_no_gradient[method]
        times[method] = sum(s.get_time() for s in res) / len(res)
        training_errors[method] = sum(s.training_error for s in res) / len(res)
        testing_errors[method] = sum(s.testing_error for s in res) / len(res)
    print("times = ", times)
    print("training_errors = ", training_errors)
    print("testing_errors = ", testing_errors)


def generate_comparison():
    m = 100      # dimension of the values
    n = 1000      # size of the data set
    c1 = 1 
    c2 = 2
    X_train, Y_train, X_test, Y_test, cov = newGenerateCompleteData(m, n, c1, c2)
    max_epochs = 50
    precision = 10**-5
    l = 2
    C = 1/(2*l*n)
    #l = 1/(2*C*n)
    eta = 0.00001
    
    compare(X_train, Y_train, X_test, Y_test, max_epochs, precision, eta, C, l)
    return

    keep_gradient = True
    my_SGD = lambda f, dim: SGD(f, dim, eta=eta, max_epochs=max_epochs, precision=precision, keep_gradient=keep_gradient)
    my_SVRG = lambda f, dim: SVRG(f, dim, m=2*n, eta=eta, max_epochs=max_epochs, precision=precision)
    my_SAGA = lambda f, dim: SAGA(f, dim, eta=eta, max_epochs=max_epochs, precision=precision, keep_gradient=keep_gradient)
    my_SDCA = lambda: SDCA(X=X_train, Y=Y_train, C=C, max_epochs=max_epochs, precision=precision, keep_Q=False, keep_gradient=keep_gradient)

    methods = {"sgd": my_SGD, "svrg": my_SVRG, "saga": my_SAGA, "sdca": my_SDCA}
    methods_list = ["sgd", "svrg", "saga", "sdca"]
    method = methods_list[0]
    if method == "sdca":
        solver = SDCASolver(X_train, Y_train, my_SDCA)
    else:
        solver = LogisticRegressionSolver(X_train, Y_train, methods[method], l=l)
    solver.solve()
    stats: Statistics = solver.get_stats()
    plt.yscale("log")
    stats.plot_gradient_norm()
    plt.show()


if __name__ == "__main__":
    generate_comparison()
