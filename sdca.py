from function import *
import random
from optimization_method import *
from common import *


class SDCA(OptimizationMethod):
    """
    Stochastic Gradient Descent
    """
    def __init__(self, f: FiniteSumFunction, dim: int, l: int = 0, eta: float = 1.0, max_epochs: int = INF, precision: float = 0.0):
        super().__init__(f, dim)
        self.l = l      # regularization parameter
        self.alpha: np.array = np.zeros(dim, dtype=DTYPE)   # solution to the dual problem
        self.eta = eta      # the learning rate
        self.n = len(f)     # size of the data set
        self.f_conj: Function = f.convex_conjugate()
        self.max_epochs = max_epochs
        self.precision = precision
        self.current_gradient = self.f_conj.gradient(self.alpha)
        self.statistics.gradient_norms.append(sq_norm(self.current_gradient))

    def step(self, i):
        super().count_step()
        grad = self.f_conj.gradient(self.alpha)
        if grad == 0:
            return
        dir = np.zeros(self.dim, dtype=DTYPE)
        dir[i] -= grad
        eta = line_search(self.f_conj, self.alpha, dir, eta=self.eta)
        self.alpha[i] -= eta*grad

    def epoch(self):
        for i in range(self.n):
            self.step(i)
        self.current_gradient = self.f_conj.gradient(self.alpha)
        self.count_epoch(sq_norm(self.current_gradient))

    def solve(self):
        super().solve()
        self.w = 1/(self.l * self.n)

    def stop_condition(self):
        return self.statistics.epoch_count >= self.max_epochs or sq_norm(self.current_gradient) <= self.precision**2
    
    def __repr__(self):
        return f"SDCA with η : {self.eta}"
