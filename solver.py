from optimization_method import OptimizationMethod
from sdca import SDCA
import numpy as np
from function import *
from typing import Type


class Solver:
    def __init__(self, X: np.ndarray, Y: np.ndarray, method: Type[OptimizationMethod], l: float = 0.0):
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

    def get_w(self) -> np.ndarray:
        return self.solver.w

    def get_grad(self) -> np.ndarray:
        return self.solver.f.gradient(self.solver.w)

    def get_stats(self) -> Statistics:
        return self.solver.statistics
    
    def error(self, X: np.ndarray, Y: np.ndarray) -> float:
        return 0.0


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
    
    def probability(self, x: np.ndarray, y: int, w: np.ndarray=None):
        if w is None:
            w = self.solver.w
        return 1 / (1 + np.exp(-y * w @ x))
    
    def predicted_label(self, x: np.ndarray, w: np.ndarray=None):
        return 1 if self.probability(x, 1, w) > 0.5 else -1
    
    def error(self, X, Y, w=None):
        absolute_error = sum(self.predicted_label(x, w) != y for (x, y) in zip(X, Y))
        return absolute_error / len(Y)
        errors = [0, 0]
        for (x, y) in zip(X, Y):
            errors[0 if y == 1 else 1] += (self.predicted_label(x) != y)
        errors[0] /= np.count_nonzero(Y+1)
        errors[1] /= np.count_nonzero(Y-1)
        return errors
    
    def training_error(self, w=None):
        return self.error(self.X, self.Y, w)
    

class SDCASolver(LogisticRegressionSolver):
    def create_solver(self):
        self.solver = self.method()
