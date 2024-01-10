
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ortho_group

def covMatrix(dim,c1,c2):
    ''' Makes a covariance matrix of size dim, with correlation parameter corr
    Args:
        dim: Int, number of dimensions of the data to generate
        c1: Float, positive number
        c2:
    Returns:
        (dim, dim) covariance matrix as numpy array
    '''
    V = ortho_group.rvs(dim=dim) # Makes a matrix of linearly independent vectors to use as eigenvectors
    L = np.diag(np.arange(1,dim+1)*c2)**c1 / c2 # Makes a diagonal matrix of ordered values to use as eigenvalues, multiplies them by c2 to uniformly the variances along the eigenvectors. To the power of c2 to make the data more correlated with some eigenvectors than others
    return V @ L @ np.linalg.inv(V), V, L # Assembles and the covariance matrix from its eigendecomposition, as well as V and L

def is_pos_def(x):
    if np.all(np.linalg.eigvals(x) >= 0) == False:
        raise Exception('The covariance matrix is not positive semi-definite')

def mulitnormalRandomData(n_samples, dim, c1, c2):
    ''' Generates random data
    Args:
        n_samples: Int, the number of data points to draw
        dim: Int, dimension of the dataspace
        c1: Float, positive number
        c2:
    Returns:
        data: (n_samples,dim) array of the drawn data points
        Sigma: The covariance matrix of the generated data
    '''
    Sigma, V, L = covMatrix(dim, c1, c2)
    is_pos_def(Sigma)

    data = np.random.multivariate_normal(np.zeros(dim), Sigma, n_samples)
    return data, Sigma, V, L
    
def display_data(data, eigVecs, eigVals):
    fig, ax = plt.subplots()

    ax.scatter(data[:,0],data[:,1])

    origin = np.zeros(2)
    ax.quiver(origin, origin, eigVecs[0,:] * eigVals, eigVecs[1,:] * eigVals)
    plt.show()


def testDataGeneration():
    data, cov, V, L = mulitnormalRandomData(1000,2,3,2)
    eVals = np.diag(L)

    print(cov)

    display_data(data, V, eVals)


def probability(x: np.ndarray, y: int, w: np.ndarray):
    return 1 / (1 + np.exp(-y * w @ x))


def predicted_label(x: np.ndarray, w: np.ndarray):
    return 1 if probability(x, 1, w) > 0.5 else -1

def generateLabels(X, w, sigma=0.1):
    rng = np.random.default_rng()
    dim = X.shape[1]
    return np.array([predicted_label(x, w + sigma*rng.standard_normal(dim)) for x in X])


def newGenerateCompleteData(dim, nb_samples, c1, c2, w=None):
    X, cov, V, L = mulitnormalRandomData(nb_samples, dim, c1, c2)
    if w is None:
        w = np.random.rand(dim)
    Y = generateLabels(X, w)
    test_index = nb_samples - nb_samples // 10
    X_train, Y_train = X[:test_index], Y[:test_index]
    X_test, Y_test = X[test_index:], Y[test_index:]
    return X_train, Y_train, X_test, Y_test, cov

