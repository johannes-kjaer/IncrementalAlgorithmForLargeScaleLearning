import time
from function import Function
import numpy as np
import matplotlib.pyplot as plt
from common import *


class OptimizationMethod:
    def __init__(self, f: Function, dim: int):
        self.f = f          # function to optimize
        self.dim = dim      # dimension of the values
        self.statistics = Statistics()
        self.w = np.zeros(dim, dtype=DTYPE)     # solution of the optimization

    def count_step(self):
        self.statistics.step()

    def step(self, i: int):
        pass

    def count_epoch(self, gradient_norm: float = 0.0):
        self.statistics.epoch(gradient_norm)

    def epoch(self):
        pass

    def stop_condition(self):
        return True

    def solve(self):
        self.statistics.start()
        while not self.stop_condition():
            self.epoch()
        self.statistics.stop()
