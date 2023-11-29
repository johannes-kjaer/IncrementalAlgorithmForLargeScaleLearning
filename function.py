from typing import List, Type
import numpy as np
from common import *


class Function:
    def evaluate(self, w: np.array) -> DTYPE:
        return DTYPE(0)

    def __call__(self, w: np.array):
        return self.evaluate(w)

    def gradient(self, w: np.array) -> np.array:
        return np.array([], dtype=DTYPE)

    def __add__(self, other):
        class Sum(Function):
            def evaluate(s, w):
                return self.evaluate(w) + other.evaluate(w)
            def gradient(s, w):
                return self.gradient(w) + other.gradient(w)
        return Sum()

    def __mul__(self, other):
        class Prod(Function):
            def evaluate(s, w):
                if isinstance(other, Function):
                    return self.evaluate(w) * other.evaluate(w)
                else:
                    return self.evaluate(w) * other
            def gradient(s, w):
                if isinstance(other, Function):
                    return self.gradient(w) * other.evaluate(w) + self.evaluate(w) * other.gradient(w)
                else:
                    return self.gradient(w) * other
        return Prod()

    def __rmul__(self, other):
        return self.__mul__(other)

    @classmethod
    def make(cls, evaluate, gradient):
        f = cls()
        f.evaluate = evaluate
        f.gradient = gradient
        return f


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


class QuadraticSum(FiniteSumFunction):
    """
    of the form 1/n * ||Xw - Y||^2
    """
    def __init__(self, X: np.array, Y: np.array):
        self.X = X
        self.Y = Y
        components = [Quadratic(x, y) for (x, y) in zip(self.X, self.Y)]
        super().__init__(components)

    def evaluate(self, w: np.array) -> DTYPE:
        return 1/self.n * sq_norm(self.X @ w - self.Y)

    def gradient(self, w: np.array):
        return 1/self.n * 2 * (self.X @ w - self.Y) @ self.X


class LogLikelihood(Function):
    """
    of the form log(1 + exp(-y(w.x)))
    """
    def __init__(self, x: np.array, y: int):
        self.x = x
        self.y = y
    
    def evaluate(self, w: np.array) -> DTYPE:
        return np.log(1 + np.exp(-self.y * w @ self.x))
    
    def gradient(self, w: np.array):
        exp = np.exp(-self.y * w @ self.x)
        return -self.y * self.x * exp / (1 + exp)


def regularize(FuncType: Type[Function]):
    class RegularizedFunction(FuncType):
        def __init__(self, *args, **kwargs):
            if "l" not in kwargs:
                raise KeyError("Regularized function constructor must get an 'l' keyword argument")
            self.l = kwargs["l"]
            kwargs.pop("l")
            super().__init__(*args, **kwargs)

        def evaluate(self, w: np.array) -> DTYPE:
            return super().evaluate(w) + self.l * sq_norm(w)

        def gradient(self, w: np.array) -> np.array:
            return super().gradient(w) + 2 * self.l * w

        def __getitem__(self, item):
            if not isinstance(self, FiniteSumFunction):
                raise TypeError(f"{repr(FuncType)} object is not subscriptable")
            return super().__getitem__(item) + self.l * Function.make(evaluate=sq_norm, gradient=lambda w: 2*w)
    return RegularizedFunction


RegularizedQuadratic = regularize(Quadratic)


class DummyFunction(Function):
    def evaluate(self, w: np.array):
        return DTYPE(0)
    
    def gradient(self, w: np.array):
        return np.zeros(w.shape, dtype=DTYPE)


class DummySumFunction(FiniteSumFunction):
    def __init__(self, n: int = 2):
        components = [DummyFunction() for _ in range(n)]
        super().__init__(components)
