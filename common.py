import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Callable


INF = 10**100
DTYPE = np.float32


def sq_norm(x: np.array):
    return x @ x


def line_search(f: Callable, w: np.array, dir: np.array, eta: float = 1.0, base: float = 2, max_depth: int = 20):
    f0 = f(w)
    depth = 0
    while depth < max_depth and f(w+eta*dir) >= f0:
        depth += 1
        eta /= base
    return 0 if depth == max_depth else eta


class Statistics:
    def __init__(self, method):
        self.method = method
        self.step_count = 0
        self.epoch_count = 0
        self.start_time = 0.0
        self.stop_time = 0.0
        self.gradient_norms = []
        self.times = []

    def start(self):
        self.start_time = time.time()
        print(f"Epoch {self.epoch_count}", end="")

    def stop(self):
        print()
        self.stop_time = time.time()

    def step(self):
        self.step_count += 1

    def epoch(self, gradient_norm: float):
        self.epoch_count += 1
        self.gradient_norms.append(gradient_norm)
        self.times.append(time.time())
        print("\r", end="")
        print(f"Epoch {self.epoch_count}", end="")

    def plot_gradient_norm(self):
        plt.plot(list(range(self.epoch_count+1)), self.gradient_norms)

    def plot_duration(self):
        durations = [self.times[i+1]-self.times[i] for i in range(self.epoch_count-1)]
        plt.plot(list(range(self.epoch_count-1)), durations)

    def __repr__(self):
        return f"{self.epoch_count} epochs in {self.stop_time - self.start_time : 3f} seconds, using {repr(self.method)}"
