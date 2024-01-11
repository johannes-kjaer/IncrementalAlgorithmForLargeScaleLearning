import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
#from scipy.stats import ortho_group


def generateRandomCovMatrix(n):
    rng = np.random.default_rng()
    A = rng.standard_normal((n, n))
    #A = np.random.rand(n, n)
    return A @ A.T

def generateRandomCovMAtrix2(dim):
    rng = np.random.default_rng()
    
    Sigma = np.diag(rng.uniform(low=0.0, high=10.0, size=dim))
    U = ortho_group.rvs(dim=dim)
    cov = U @ Sigma @ U.T

    return cov

def generateRandomData2(dim, n_samples, corr = 0.1):
    makeCorrelated = corr * np.ones((dim,dim))
    makeCorrelated += (1-corr) * np.identity(dim)

    uncorrdata = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), n_samples)

    corrData = makeCorrelated @ uncorrdata

    return corrData



def generateRandomData(dim, nb_samples, ratio=0.0, mean=None, cov=None):
    ''' Generates some random data with either zero or otherwise specified mean, with either a random covariance matrix (with the possibility of making it more or les correleated) or a specified one.
    Args:
        n: The number of variables
        samples: The number of samples to be generated
        ratio: The ratio between the random covariance and the identity matrix
        mean: (n) array of the mean of the distribution
        cov: (n,n) array of the covariance of the distribution
    Returns:
        data: (n,samples) array of the randomly generated data
        mean: (n) array of the mean of the distribution
        cov: (n,n) array of the covariance of the distribution
    '''
    if mean is None:
        mean = np.zeros(dim, dtype=np.double)

    if cov is None:
        cov = generateRandomCovMatrix(dim) + ratio * np.identity(dim, dtype=np.double)

    randomData = np.random.multivariate_normal(mean, cov, nb_samples)

    return randomData, cov, mean


def probability(x: np.ndarray, y: int, w: np.ndarray):
    return 1 / (1 + np.exp(-y * w @ x))


def predicted_label(x: np.ndarray, w: np.ndarray):
    return 1 if probability(x, 1, w) > 0.5 else -1


def generateLabels(X: np.ndarray, w: np.ndarray, sigma=0.1):
    """sigma is the amount of noise"""
    rng = np.random.default_rng()
    dim = X.shape[1]
    return np.array([predicted_label(x, w + sigma*rng.standard_normal(dim)) for x in X])


def generateCompleteData(dim: int, nb_samples: int, w: np.ndarray=None, sample_generator: Callable=None, label_generator: Callable=None, **kwargs):
    if sample_generator is None:
        sample_generator = lambda dim, nb_samples: np.random.rand(nb_samples, dim)
    if label_generator is None:
        label_generator = generateLabels
    data = sample_generator(dim, nb_samples, **kwargs)
    if isinstance(data, tuple):
        X = data[0]
        additional_data = data[1:]
    else:
        X = data
        additional_data = tuple()
    if w is None:
        w = np.random.rand(dim)
    Y = label_generator(X, w)
    test_index = nb_samples - nb_samples // 10
    X_train, Y_train = X[:test_index], Y[:test_index]
    X_test, Y_test = X[test_index:], Y[test_index:]
    return X_train, Y_train, X_test, Y_test, w, additional_data


def generateCompleteData_old(dim, nb_samples, ratio=0.0, mean=None, cov=None, w=None):
    X, cov, mean = generateRandomData(dim, nb_samples, ratio, mean, cov)
    if w is None:
        w = np.random.rand(dim)
    Y = generateLabels(X, w)
    test_index = nb_samples - nb_samples // 10
    X_train, Y_train = X[:test_index], Y[:test_index]
    X_test, Y_test = X[test_index:], Y[test_index:]
    return X_train, Y_train, X_test, Y_test, cov, mean


def generateCompleteData2(dim, nb_samples, corr =0.1, w=None):
    X = generateRandomData2(dim, nb_samples, corr)
    if w is None:
        w = np.random.rand(dim)
    Y = generateLabels(X, w)
    test_index = nb_samples - nb_samples // 10
    X_train, Y_train = X[:test_index], Y[:test_index]
    X_test, Y_test = X[test_index:], Y[test_index:]
    return X_train, Y_train, X_test, Y_test
