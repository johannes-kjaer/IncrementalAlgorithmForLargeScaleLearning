from function import *
import random
from optimization_method import *
from common import *
import math


class SGD(OptimizationMethod):
    """
    Stochastic Gradient Descent
    """
    def __init__(self, f: FiniteSumFunction, dim: int, eta: float = 1.0, max_epochs: int = INF, precision: float = 0.0, keep_gradient: bool=True):
        super().__init__(f, dim, max_epochs, precision, keep_gradient)
        self.f: FiniteSumFunction
        self.eta = eta      # the learning rate
        self.n = len(f)     # size of the data set
        self.start()

    def step(self, i):
        self.count_step()
        eta = self.eta / math.sqrt(self.statistics.step_count)
        self.w = self.w - eta * self.f[i].gradient(self.w)

    def epoch(self):
        # random_seq = np.random.permutation(self.n)
        # for i in random_seq:
        for _ in range(self.n):
            i = random.randint(0, self.n-1)
            self.step(i)
        self.current_gradient = self.get_gradient()
        self.count_epoch()

    def __repr__(self):
        if hasattr(self.f[0], "l"):
            l = self.f[0].l
        else:
            l = 0
        return f"SGD with η = {self.eta}, λ = {l}"
