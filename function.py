from typing import List
import numpy as np
from common import *


class Function:
    def evaluate(self, w: np.array) -> DTYPE:
        return DTYPE(0)

    def __call__(self, w: np.array):
        return self.evaluate(w)

    def gradient(self, w: np.array) -> np.array:
        return np.array([], dtype=DTYPE)


class FiniteSumFunction(Function):
    def __init__(self, components: List[Function]):
        super().__init__()
        self.components = components
        self.n = len(self.components)

    def evaluate(self, w: np.array) -> DTYPE:
        return DTYPE(1/self.n * sum(f.evaluate(w) for f in self.components))

    def gradient(self, w: np.array) -> np.array:
        return 1/self.n * sum(f.gradient(w) for f in self.components)

    def __getitem__(self, i: int):
        return self.components[i]

    def __len__(self):
        return len(self.components)


class Quadratic(Function):
    """
    of the form (w.x - y)^2
    """
    def __init__(self, x: np.array, y: DTYPE):
        self.x = x
        self.y = y

    def evaluate(self, w: np.array) -> DTYPE:
        return (w @ self.x - self.y)**2

    def gradient(self, w: np.array) -> np.array:
        return 2 * (w @ self.x - self.y) * self.x
