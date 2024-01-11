from function import *
import random
from optimization_method import *
from common import *


class GD(OptimizationMethod):
    """
    Stochastic Gradient Descent
    """
    def __init__(self, f: FiniteSumFunction, dim: int, eta: float = 1.0, max_epochs: int = INF, precision: float = 0.0, keep_gradient: bool=True):
        super().__init__(f, dim, max_epochs, precision, keep_gradient)
        self.f: FiniteSumFunction
        self.eta = eta      # the learning rate
        self.n = len(f)     # size of the data set
        self.start()

    def epoch(self):
        self.w -= self.eta * self.current_gradient
        self.current_gradient = self.f.gradient(self.w)
        self.count_epoch()
    
    def __repr__(self):
        if hasattr(self.f[0], "l"):
            l = self.f[0].l
        else:
            l = 0
        return f"GD with η = {self.eta}, λ = {l}"
