import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Callable
import sys


INF = 10**100
DTYPE = np.float64


def sq_norm(x: np.array):
    return x @ x


def get_method_from_user_input():
    method = "gd" if len(sys.argv) < 2 else sys.argv[1]
    return method


def line_search(f: Callable, w: np.array, dir: np.array, eta: float = 1.0, base: float = 2, max_depth: int = 20):
    f0 = f(w)
    depth = 0
    while depth < max_depth and f(w+eta*dir) >= f0:
        depth += 1
        eta /= base
    return 0 if depth == max_depth else eta


class Statistics:
    EPOCH_PRINT_PREFIX = ""

    def __init__(self, method):
        self.method = method
        self.step_count = 0
        self.epoch_count = 0
        self.start_time = 0.0
        self.stop_time = 0.0
        self.gradient_norms = []
        self.objective_values = []
        self.times = []
        self.training_error = 0.0
        self.testing_error = 0.0

    def start(self):
        self.start_time = time.time()
        print(f"{Statistics.EPOCH_PRINT_PREFIX}Epoch {self.epoch_count}", end="", flush=True)

    def stop(self):
        print()
        self.stop_time = time.time()

    def get_time(self):
        return self.stop_time - self.start_time
    
    def step(self):
        self.step_count += 1

    def epoch(self, gradient_norm: float, objective_value: float):
        self.epoch_count += 1
        self.gradient_norms.append(gradient_norm)
        self.objective_values.append(objective_value)
        self.times.append(time.time())
        print("\r", end="")
        print(f"{Statistics.EPOCH_PRINT_PREFIX}Epoch {self.epoch_count}", end="", flush=True)

    def plot_gradient_norm(self):
        plt.plot(list(range(self.epoch_count+1)), self.gradient_norms)
        plt.xlabel("Epochs")
        plt.ylabel("Gradient squared norm")
    
    def plot_objective_function(self):
        plt.plot(list(range(self.epoch_count+1)), self.objective_values)
        plt.xlabel("Epochs")
        plt.ylabel("Objective function")

    def plot_duration(self):
        durations = [self.times[i+1]-self.times[i] for i in range(self.epoch_count-1)]
        plt.plot(list(range(self.epoch_count-1)), durations)

    def __repr__(self):
        return f"{self.epoch_count} epochs in {self.stop_time - self.start_time : 3f} seconds, using {repr(self.method)}"
