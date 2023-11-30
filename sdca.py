from function import *
import random
from optimization_method import *
from common import *


class SDCA(OptimizationMethod):
    """
    SDCA algorithm for the particular case of logistic regression in the context of 'maxtent' paper. Algorithm 5 of the paper
    primal of the form :
    min p(w) = C * sum_{i=1 to n} log(1 + exp(-yᵢwᵀxᵢ)) + ½║w║²
    primal :
                n
               \‾‾
    min P(w) = /__ log(1 + exp(-yᵢwᵀxᵢ)) + ½║w║²
     ʷ         i=1


    dual :

                       \‾‾             \‾‾
    min D(α) = ½αᵀQα + /__ αᵢlog(αᵢ) + /__ (C - αᵢ)log(C - αᵢ)
     ᵅ                i:αᵢ>0          i:αᵢ<C
    subject to 0 ≤ αᵢ ≤ C
    where Qᵢⱼ = yᵢyⱼxᵢᵀxⱼ
    (wasted way too much time formating this)
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, dim: int, C: float = 0.0, eta: float = 1.0, max_epochs: int = INF, precision: float = 0.0):
        self.Q = (Y[np.newaxis].T @ Y[np.newaxis]) * (X.T @ X)
        f = LogisticRegressionDual(C, self.Q)
        super().__init__(f, dim)
        self.X = X
        self.Y = Y
        self.C = C      # regularization parameter
        self.xi = 0.25  # TODO: how to choose xi ? what is xi ?
        epsilon1, epsilon2 = 0.1, 0.1   # TODO: how to choose epsilon1, epsilon2 ?
        self.alpha: np.ndarray = min(epsilon1*C, epsilon2) * np.ones(dim, dtype=DTYPE)   # solution to the dual problem
        self.alpha_prime: np.ndarray = C - self.alpha
        self.w = np.sum((self.alpha*Y) * X.T, axis=1)
        self.eta = eta      # the learning rate
        self.n = X.shape[0]     # size of the data set
        self.max_epochs = max_epochs
        self.precision = precision
        self.current_gradient = self.f.gradient(self.alpha)
        self.statistics.gradient_norms.append(sq_norm(self.current_gradient))

    def modified_newton_method(self, a, b, c1, c2):
        b_ = [b, -b]
        c_ = [c1, c2]
        s = c1 + c2
        z_m = (c2 - c1) / 2
        t = 0 if z_m >= -b/a else 1
        g_prime = lambda Zt: np.log(Zt/(s-Zt)) + a(Zt-c_[t]) + b_[t]
        g_doubleprime = lambda Zt: a + s / (Zt(s-Zt))
        Zt = s/2  # choose in (0, s) ?
        while True:
            if g_prime(Zt) == 0:
                break
            d = g_prime(Zt) / g_doubleprime(Zt)
            Zt = self.xi * Zt if Zt + d <= 0 else Zt + d
        Z = [0, 0]
        Z[t] = Zt
        Z[1-t] = s-Zt
        return Z
    
    def step(self, i):
        c1 = self.alpha[i]
        c2 = self.alpha_prime[i]
        a = self.Q[i, i]
        b = self.Y[i] * self.w @ self.X[i]
        Z1, Z2 = self.modified_newton_method(a, b, c1, c2)
        self.w += (Z1 - self.alpha[i]) * self.Y[i] * self.X[i]
        self.alpha[i] = Z1
        self.alpha_prime[i] = Z2

    def epoch(self):
        for i in range(self.n):
            self.step(i)
        self.current_gradient = self.f.gradient(self.alpha)
        self.count_epoch(sq_norm(self.current_gradient))


# class SDCA(OptimizationMethod):
#     """
#     Stochastic Gradient Descent
#     """
#     def __init__(self, f: FiniteSumFunction, dim: int, l: float = 0.0, eta: float = 1.0, max_epochs: int = INF, precision: float = 0.0):
#         super().__init__(f, dim)
#         self.l = l      # regularization parameter
#         self.alpha: np.array = np.zeros(dim, dtype=DTYPE)   # solution to the dual problem
#         self.eta = eta      # the learning rate
#         self.n = len(f)     # size of the data set
#         self.f_conj: Function = f.convex_conjugate()
#         self.max_epochs = max_epochs
#         self.precision = precision
#         self.current_gradient = self.f_conj.gradient(self.alpha)
#         self.statistics.gradient_norms.append(sq_norm(self.current_gradient))

#     def step(self, i):
#         super().count_step()
#         grad = self.f_conj.gradient(self.alpha)
#         if grad == 0:
#             return
#         dir = np.zeros(self.dim, dtype=DTYPE)
#         dir[i] -= grad
#         eta = line_search(self.f_conj, self.alpha, dir, eta=self.eta)
#         self.alpha[i] -= eta*grad

#     def epoch(self):
#         for i in range(self.n):
#             self.step(i)
#         self.current_gradient = self.f_conj.gradient(self.alpha)
#         self.count_epoch(sq_norm(self.current_gradient))

#     def solve(self):
#         super().solve()
#         self.w = 1/(self.l * self.n)

#     def stop_condition(self):
#         return self.statistics.epoch_count >= self.max_epochs or sq_norm(self.current_gradient) <= self.precision**2
    
#     def __repr__(self):
#         return f"SDCA with η : {self.eta}"
