from function import *
import random
from optimization_method import *
from common import *


class SAGA(OptimizationMethod):
    """
    Stochastic Average Gradient Augmented
    """

    def __init__(self, f: FiniteSumFunction, dim: int, eta: float = 1.0, max_epochs: int = INF, precision: float = 0.0, keep_gradient: bool=True):
        super().__init__(f, dim, max_epochs, precision, keep_gradient)
        self.eta = eta  # the learning rate
        self.n = len(f)  # size of the data set
        self.g = np.zeros((self.n, dim))
        self.grad_sum = np.zeros(dim, dtype=DTYPE)
        self.start()

    def step(self, i):
        self.count_step()
        current_sg = self.f[i].gradient(self.w)  # Computing the gradient
        self.w = self.w - self.eta * (
                    current_sg - self.g[i] + 1/self.n * self.grad_sum)#np.sum(self.g, axis=0))  # Doing on SAGA parameter update
        self.grad_sum += current_sg - self.g[i]
        self.g[i] = current_sg  # Updating the table entry of the gradient

    def epoch(self):
        # for i in range(self.n):
        #     i = random.randint(0, self.n-1)
        #     self.step(i)
        random_seq = np.random.permutation(self.n)
        for i in random_seq:
            self.step(i)
        self.current_gradient = self.get_gradient()
        self.count_epoch()
    
    def __repr__(self):
        if hasattr(self.f[0], "l"):
            l = self.f[0].l
        else:
            l = 0
        return f"SAGA with η = {self.eta}, λ = {l}"
