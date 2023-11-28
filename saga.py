from function import *
import random
from optimization_method import *
from common import *

class SAGA(OptimizationMethod):
    """
    Stochastic Average Gradient Augmented
    """
    def __init__(self, f: Function, dim: int, eta: float = 1.0, max_epochs: int = INF, precision: float = 0.0):
        super().__init__(f, dim)
        self.eta = eta      # the learning rate
        self.n = len(f)     # size of the data set
        self.max_epochs = max_epochs
        self.precision = precision
        self.current_gradient = self.f.gradient(self.w)
        self.statistics.gradient_norms.append(sq_norm(self.current_gradient))
        self.g = np.zeros((self.n,dim))

    def step(self, i):
        super().count_step()
        current_sg = self.f[i].gradient(self.w) # Computing the gradient
        self.w = self.w - self.eta * (current_sg - self.g[i,:] + np.sum(self.g,axis=0)) # Doing on SAGA parameter update
        self.g[i,:] = current_sg # Updating the table entry of the gradient
    
    def epoch(self):
        random_seq = np.random.permutation(self.n)
        for i in random_seq:
            self.step(i)
        self.current_gradient = self.f.gradient(self.w)
        self.count_epoch()
    
    def stop_condition(self):
        return self.statistics.epoch_count >= self.max_epochs or sq_norm(self.current_gradient) <= self.precision**2