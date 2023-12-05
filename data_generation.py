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
    if mean == None:
        mean = np.zeros(dim, dtype=np.double)

    if cov == None:
        cov = generateRandomCovMatrix(dim) + ratio * np.identity(dim, dtype=np.double)

    randomData = np.random.multivariate_normal(mean, cov, nb_samples)

    return randomData, cov, mean
