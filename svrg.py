from function import *
import random
from optimization_method import *
from common import *


class SVRG(OptimizationMethod):
    """
    Stochastic Variance Reduced Gradient
    """
    def __init__(self, f: FiniteSumFunction, dim: int, m: int, eta: float = 1.0, max_epochs: int = INF, precision: float = 0.0):
        super().__init__(f, dim)
        self.w_tilde = np.zeros(self.dim, dtype=DTYPE)
        self.m = m          # update frequency
        self.eta = eta      # the learning rate
        self.n = len(f)     # size of the data set
        self.max_epochs = max_epochs
        self.precision = precision
        self.mu_tilde = self.f.gradient(self.w_tilde)   # current gradient
        self.statistics.gradient_norms.append(sq_norm(self.mu_tilde))

    def step(self, i):
        super().count_step()
        self.w = self.w - self.eta * (self.f[i].gradient(self.w) - self.f[i].gradient(self.w_tilde) + self.mu_tilde)

    def epoch(self):
        self.w = self.w_tilde
        for _ in range(self.m):
            i = random.randint(0, self.n-1)
            self.step(i)
        self.w_tilde = self.w
        self.mu_tilde = self.f.gradient(self.w_tilde)
        self.count_epoch(sq_norm(self.mu_tilde))

    def stop_condition(self):
        return self.statistics.epoch_count >= self.max_epochs or sq_norm(self.mu_tilde) <= self.precision**2
    
    def __repr__(self):
        return f"SVRG with η : {self.eta}, m : {self.m}"
