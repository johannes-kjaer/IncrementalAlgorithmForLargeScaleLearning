from optimization_method import OptimizationMethod
import numpy as np
from function import *
from typing import Type


class Solver:
    def __init__(self, X: np.array, Y: np.array, method: Type[OptimizationMethod], l: float = 0.0):
        self.X = X
        self.Y = Y
        self.m = self.X.shape[1]        # dimension of the values
        self.method = method            # the method type to use, a subclass of OptimizationMethod
        self.solver: method = None      # the actual instance of this method type
        self.l = l                      # regularization parameter
        self.create_solver()

    def create_solver(self):
        pass

    def solve(self):
        self.solver.solve()

    def get_w(self) -> np.array:
        return self.solver.w

    def get_grad(self) -> np.array:
        return self.solver.f.gradient(self.solver.w)

    def get_stats(self) -> Statistics:
        return self.solver.statistics


class LinearRegressionSolver(Solver):
    def create_solver(self):
        loss_function = QuadraticSum(self.X, self.Y)
        self.solver = self.method(loss_function, self.m)


class LogisticRegressionSolver(Solver):
    def create_solver(self):
        if self.l == 0:
            loss_functions = [LogLikelihood(x, y) for (x, y) in zip(self.X, self.Y)]
        else:
            loss_functions = [RegularizedLogLikelihood(x, y, l=self.l) for (x, y) in zip(self.X, self.Y)]
        loss_function = FiniteSumFunction(loss_functions)
        self.solver = self.method(loss_function, self.m)
    
    def probability(self, x: np.array, y: np.array):
        return 1 / (1 + np.exp(-y * self.solver.w @ x))
    
    def predicted_label(self, x: np.array):
        return 1 if self.probability(x, 1) > 0.5 else -1
    
    def error(self, X, Y):
        errors = [0, 0]
        for (x, y) in zip(X, Y):
            errors[0 if y == 1 else 1] += (self.predicted_label(x) != y)
        errors[0] /= np.count_nonzero(Y+1)
        errors[1] /= np.count_nonzero(Y-1)
        return errors
    
    def training_error(self):
        return self.error(self.X, self.Y)
    
