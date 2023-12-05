from typing import List, Type
import numpy as np
from common import *


class Function:
    def evaluate(self, w: np.ndarray) -> DTYPE:
        return DTYPE(0)

    def __call__(self, w: np.ndarray):
        return self.evaluate(w)

    def gradient(self, w: np.ndarray) -> np.ndarray:
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

    def evaluate(self, w: np.ndarray) -> DTYPE:
        return DTYPE(1/self.n * sum(f.evaluate(w) for f in self.components))

    def gradient(self, w: np.ndarray) -> np.ndarray:
        return 1/self.n * sum(f.gradient(w) for f in self.components)

    def __getitem__(self, i: int):
        return self.components[i]

    def __len__(self):
        return len(self.components)


class DummyFunction(Function):
    """for testing"""
    def evaluate(self, w: np.ndarray):
        return DTYPE(0)
    
    def gradient(self, w: np.ndarray):
        return np.zeros(w.shape, dtype=DTYPE)


class DummySumFunction(FiniteSumFunction):
    """for testing"""
    def __init__(self, n: int = 2):
        components = [DummyFunction() for _ in range(n)]
        super().__init__(components)


class Quadratic(Function):
    """
    of the form (w.x - y)^2
    """
    def __init__(self, x: np.ndarray, y: DTYPE):
        self.x = x
        self.y = y

    def evaluate(self, w: np.ndarray) -> DTYPE:
        return (w @ self.x - self.y)**2

    def gradient(self, w: np.ndarray) -> np.ndarray:
        return 2 * (w @ self.x - self.y) * self.x


class QuadraticSum(FiniteSumFunction):
    """
    of the form 1/n * ||Xw - Y||^2
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X
        self.Y = Y
        components = [Quadratic(x, y) for (x, y) in zip(self.X, self.Y)]
        super().__init__(components)

    def evaluate(self, w: np.ndarray) -> DTYPE:
        return 1/self.n * sq_norm(self.X @ w - self.Y)

    def gradient(self, w: np.ndarray) -> np.ndarray:
        return 1/self.n * 2 * (self.X @ w - self.Y) @ self.X


class LogLikelihood(Function):
    """
    of the form log(1 + exp(-y(w.x)))
    """
    def __init__(self, x: np.ndarray, y: int):
        self.x = x
        self.y = y
    
    def evaluate(self, w: np.ndarray) -> DTYPE:
        return np.log(1 + np.exp(-self.y * w @ self.x))
    
    def gradient(self, w: np.ndarray):
        exp = np.exp(-self.y * w @ self.x)
        return -self.y * self.x * exp / (1 + exp)


class LogisticRegressionPrimal(Function):
    """
    of the form
    """
    def __init__(self, X, Y, C: float):
        self.X = X
        self.Y = Y
        self.C = C
    
    def evaluate(self, w):
        return self.C * np.sum(np.log(1 + np.exp(-self.Y * self.X@w))) + 0.5 * sq_norm(w)
    
    def gradient(self, w):
        exp = np.exp(-self.Y * self.X@w)
        return self.C * np.sum(1/(1+exp)*exp * (-self.Y*self.X), axis=1)


class LogisticRegressionDual(Function):
    """
    From max'maxent' paper, of the form
            \‾‾             \‾‾
    ½αᵀQα + /__ αᵢlog(αᵢ) + /__ (C - αᵢ)log(C - αᵢ)
           i:αᵢ>0          i:αᵢ<C
    """
    def __init__(self, C: float, Q: np.ndarray):
        self.C = C
        self.Q = Q
    
    @staticmethod
    def extract_positive(v: np.ndarray, default: DTYPE = 1):
        """return an array w with w[i] = v[i] if v[i] > 0 else default"""
        return (v > 0)*v + (v <= 0)*np.ones(len(v))*default
    
    def evaluate(self, alpha):
        positive_alpha = self.extract_positive(alpha)
        less_than_C_alpha = self.extract_positive(self.C-alpha)
        return 0.5 * alpha @ self.Q @ alpha + np.sum(alpha * np.log(positive_alpha)) + np.sum((self.C - alpha) * np.log(less_than_C_alpha))
    
    def gradient(self, alpha):
        positive_alpha = self.extract_positive(alpha)
        less_than_C_alpha = self.extract_positive(self.C-alpha)
        return self.Q @ alpha + (np.log(positive_alpha) + 1) * (alpha > 0) - (np.log(less_than_C_alpha) + 1) * (alpha < self.C)


def regularized(FuncType: Type[Function]):
    class RegularizedFunction(FuncType):
        """
        Of the form w --> f(w) + l||w||^2.
        The constructor must get an 'l' keyword (otherwise a non regularized function is returned)
        """
        def __new__(cls, *args, **kwargs):
            if "l" not in kwargs or kwargs["l"] == 0:
                if "l" in kwargs: kwargs.pop("l")
                return FuncType(*args, **kwargs)
            return super().__new__(cls)
        
        def __init__(self, *args, **kwargs):
            self.l = kwargs["l"]
            kwargs.pop("l")
            super().__init__(*args, **kwargs)

        def evaluate(self, w: np.ndarray) -> DTYPE:
            return super().evaluate(w) + self.l * sq_norm(w)

        def gradient(self, w: np.ndarray) -> np.ndarray:
            return super().gradient(w) + 2 * self.l * w

        def __getitem__(self, item):
            if not isinstance(self, FiniteSumFunction):
                raise TypeError(f"{repr(FuncType)} object is not subscriptable")
            return super().__getitem__(item) + self.l * Function.make(evaluate=sq_norm, gradient=lambda w: 2*w)
    return RegularizedFunction


RegularizedQuadratic = regularized(Quadratic)
RegularizedQuadraticSum = regularized(QuadraticSum)
RegularizedLogLikelihood = regularized(LogLikelihood)
