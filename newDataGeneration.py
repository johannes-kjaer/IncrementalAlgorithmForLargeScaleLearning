
import numpy as np
from scipy.stats import ortho_group

def covMatrix(dim,corr):
    ''' Makes a covariance matrix of size dim, with correlation parameter corr
    Args:
        dim: Int, number of dimensions of the data to generate
        corr: Float, positive number. If zero the data is uncorrelated, the larger the value, the more correlated the data gets.
    Returns:
        (dim, dim) covariance matrix array'''
    V = ortho_group.rvs(dim=dim) # Makes a matrix of linearly independent vectors to use as eigenvectors
    L = np.diag(1,np.arange(dim+1))**corr # Makes a diagonal matrix of ordered values to use as eigevalues. To the power of corr to make the axes more or less correlated
    return V @ L @ np.inverse(V) # Assembles the covariance matrix from its eigendecomposition

def is_pos_def(x):
    if np.all(np.linalg.eigvals(x) >= 0) == False:
        raise Exception('The covariance matrix is not positive semi-definite')

def newDataGeneration(n_samples, dim, corr):
    ''' Generates random data
    Args:
        n_samples: Int, the number of data points to draw
        dim: Int, dimension of the dataspace
        corr: Float, positive number. Zero gives uncorrelated data, large values gives higly correlated data.
    Returns:
        data: (n_samples,dim) array of the drawn data points
        Sigma: The covariance matrix of the generated data
    '''
    Sigma = covMatrix(dim, corr)
    is_pos_def(Sigma)

    data = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), n_samples)
    return data, Sigma

    

