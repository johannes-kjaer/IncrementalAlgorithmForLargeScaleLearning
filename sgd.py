from function import *
import random
from optimization_method import *
from common import *


class SGD(OptimizationMethod):
    """
    Stochastic Gradient Descent
    """
    def __init__(self, f: FiniteSumFunction, dim: int, eta: float = 1.0, max_epochs: int = INF, precision: float = 0.0):
        super().__init__(f, dim)
        self.eta = eta      # the learning rate
        self.n = len(f)     # size of the data set
        self.max_epochs = max_epochs
        self.precision = precision
        self.current_gradient = self.f.gradient(self.w)
        self.statistics.gradient_norms.append(sq_norm(self.current_gradient))

    def step(self, i):
        super().count_step()
        eta = self.eta  # / self.statistics.step_count
        self.w = self.w - eta * self.f[i].gradient(self.w)

    def epoch(self):
        ordering = list(range(self.n))
        random.shuffle(ordering)
        for i in ordering:
            self.step(i)
        self.current_gradient = self.f.gradient(self.w)
        self.count_epoch(sq_norm(self.current_gradient))

    def stop_condition(self):
        return self.statistics.epoch_count >= self.max_epochs or sq_norm(self.current_gradient) <= self.precision**2
