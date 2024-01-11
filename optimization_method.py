import time
from function import Function
import numpy as np
import matplotlib.pyplot as plt
from common import *


class OptimizationMethod:
    def __init__(self, f: Function, dim: int, max_epochs: int = INF, precision: float = 0.0, keep_gradient: bool=True):
        self.f = f          # function to optimize
        self.dim = dim      # dimension of the values
        self.statistics = Statistics(self)
        self.keep_gradient = keep_gradient
        self.max_epochs = max_epochs
        self.precision = precision
        self.w = np.zeros(dim, dtype=DTYPE)     # solution of the optimization
        self.current_gradient = np.zeros(dim, dtype=DTYPE)

    def start(self, point: np.ndarray=None):
        if point is None:
            point = self.w
        self.current_gradient = self.get_gradient(point)
        self.statistics.gradient_norms.append(sq_norm(self.current_gradient))
        self.statistics.objective_values.append(self.f(point))

    def count_step(self):
        self.statistics.step()

    def step(self, i: int):
        pass

    def count_epoch(self, point: np.ndarray=None):
        if point is None:
            point = self.w
        gradient_norm = sq_norm(self.current_gradient)
        objective_value = self.f(point)
        self.statistics.epoch(gradient_norm, objective_value)

    def epoch(self):
        pass

    def get_gradient(self, point: np.ndarray=None):
        if point is None:
            point = self.w
        return np.array([100.0]) if not self.keep_gradient else self.f.gradient(point)
    
    def stop_condition(self):
        return self.statistics.epoch_count >= self.max_epochs or sq_norm(self.current_gradient) <= self.precision**2
    
    def solve(self):
        self.statistics.start()
        while not self.stop_condition():
            self.epoch()
        self.statistics.stop()
