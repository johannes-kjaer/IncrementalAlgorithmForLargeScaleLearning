from optimization_method import OptimizationMethod
import numpy as np
from function import *
from typing import Type


class Solver:
    def __init__(self, method: Type[OptimizationMethod]):
        self.method = method            # the method type to use, a subclass of OptimizationMethod
        self.solver: method = None      # the actual instance of this method type
        self.create_solver()

    def create_solver(self):
        pass

    def solve(self):
        self.solver.solve()

    def get_sol(self):
        return self.solver.w

    def get_grad(self):
        return self.solver.f.gradient(self.solver.w)

    def get_stats(self):
        return self.solver.statistics


class LinearRegressionSolver(Solver):
    def __init__(self, X: np.array, Y: np.array, method: Type[OptimizationMethod]):
        self.X = X
        self.Y = Y
        self.m = self.X.shape[1]    # dimension of the values
        super().__init__(method)

    def create_solver(self):
        loss_function = QuadraticSum(self.X, self.Y)
        self.solver = self.method(loss_function, self.m)
