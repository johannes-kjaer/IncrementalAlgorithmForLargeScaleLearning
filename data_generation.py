import numpy as np
import matplotlib.pyplot as plt


def generateRandomCovMatrix(n):
    rng = np.random.default_rng()
    A = rng.standard_normal((n, n))  
    #A = np.random.rand(n, n)
    return A @ A.T


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


def generateLabels(X, w, sigma=0.1):
    rng = np.random.default_rng()
    dim = X.shape[1]
    return np.array([predicted_label(x, w + sigma*rng.standard_normal(dim)) for x in X])


def generateCompleteData(dim, nb_samples, ratio=0.0, mean=None, cov=None, w=None):
    X, cov, mean = generateRandomData(dim, nb_samples, ratio, mean, cov)
    if w is None:
        w = np.random.rand(dim)
    Y = generateLabels(X, w)
    test_index = nb_samples - nb_samples // 10
    X_train, Y_train = X[:test_index], Y[:test_index]
    X_test, Y_test = X[test_index:], Y[test_index:]
    return X_train, Y_train, X_test, Y_test, cov, mean
