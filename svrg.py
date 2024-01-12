from function import *
import random
from optimization_method import *
from common import *


class SVRG(OptimizationMethod):
    """
    Stochastic Variance Reduced Gradient
    """
    def __init__(self, f: FiniteSumFunction, dim: int, m_coef: float, eta: float = 1.0, max_epochs: int = INF, precision: float = 0.0):
        super().__init__(f, dim, max_epochs, precision, True)
        self.f: FiniteSumFunction
        self.w_tilde = np.zeros(self.dim, dtype=DTYPE)
        self.eta = eta      # the learning rate
        self.n = len(f)     # size of the data set
        if m_coef >= self.n:
            raise ValueError(f"SVRG m_coef is too big, it should be the ration m/n")
        self.m = int(m_coef * self.n)          # update frequency
        
        self.start(self.w_tilde)
        self.mu_tilde = self.current_gradient

    def step(self, i):
        self.count_step()
        self.w = self.w - self.eta * (self.f[i].gradient(self.w) - self.f[i].gradient(self.w_tilde) + self.mu_tilde)

    def epoch(self):
        self.w = self.w_tilde
        for _ in range(self.m):
            i = random.randint(0, self.n-1)
            self.step(i)
        self.w_tilde = self.w
        self.mu_tilde = self.get_gradient(self.w_tilde)
        self.current_gradient = self.mu_tilde
        self.count_epoch()
    
    def __repr__(self):
        if hasattr(self.f[0], "l"):
            l = self.f[0].l
        else:
            l = 0
        return f"SVRG with η = {self.eta}, m = {self.m}, λ = {l}"
