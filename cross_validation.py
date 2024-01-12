from typing import Callable, Any, Tuple
from solver import *
from sgd import *
from gd import *
from svrg import *
from saga import *
from sdca import *
import numpy as np


SolverTemplate = Callable[[np.ndarray, np.ndarray, Any], Solver]


def subdivise(S: np.ndarray, K: int) -> list:
    n = S.shape[0]
    block_size = n // K
    return [S[i*block_size : n if i == K-1 else (i+1)*block_size] for i in range(K)]


def solve_on_blocks(solver_template: SolverTemplate, Xblocks: list, Yblocks: list, value: Any) -> Tuple[float, float]:
    errors = []
    epoch_counts = []
    for i, (X_test, Y_test) in enumerate(zip(Xblocks, Yblocks)):
        Statistics.EPOCH_PRINT_PREFIX = f"\tValidating with block {i}: "
        X_train = np.vstack([Xblocks[j] for j in range(len(Xblocks)) if j != i])
        Y_train = np.hstack([Yblocks[j] for j in range(len(Yblocks)) if j != i])
        solver: Solver = solver_template(X_train, Y_train, value)
        solver.solve()
        errors.append(solver.error(X_test, Y_test))
        epoch_counts.append(solver.get_stats().epoch_count)
    Statistics.EPOCH_PRINT_PREFIX = ""
    K = len(Xblocks)
    return 1/K * sum(errors), 1/K * sum(epoch_counts)


def cross_validation(X: np.ndarray, Y: np.ndarray, K: int, solver_template: SolverTemplate, param_values: list, param_name: str="unknown_param") -> Solver:
    Xblocks = subdivise(X, K)
    Yblocks = subdivise(Y, K)
    errors = []
    epoch_counts = []
    for v in param_values:
        print(f"Training with {param_name} = {v}")
        try:
            error, epoch_count = solve_on_blocks(solver_template, Xblocks, Yblocks, v)
        except Exception as e:
            error, epoch_count = INF, INF
            print(f"Warning: runtime error when cross validating for value {v}")
        errors.append(error)
        epoch_counts.append(epoch_count)
    print()
    for error, value in zip(errors, param_values):
        print(f"Error of {error} with {param_name} = {value}.")
    # find the best param value based on minimal error, then minimal epoch_count if equality of errors
    i = min(range(len(param_values)), key=lambda j: (errors[j], epoch_counts[j]))
    v = param_values[i]
    print(f"Selecting {param_name} = {v}")
    return v


def get_logistic_regression_solver_template(max_epochs: int, precision: float, keep_gradient: bool, *, eta: float=None, l: float=None, C: float=None, svrg_m_coef: int=None) -> Tuple[SolverTemplate, str]:
    """
    Create solver template for cross validation.
    The possible hyperparameters for validation are
        - eta: the learning rate (for all methods except SDCA),
        - l: the normalization parameter (for all methods except SDCA),
        - C: the normalization parameter (only for SDCA),
        - svrg_m_coef: the ratio between m parameter of SVRG and n (only for SVRG).
    The hyperparameters must be given as keyword arguments to the function. 
    The one that is not specified when calling the function (or for which None is given) will be the one for which cross validation will be performed 
    (if multiple are not specified, the selected one will be the first not specified in the order eta, l, C, svrg_m_coef).
    """
    if eta is None:
        validating_param = "eta"
    elif l is None:
        validating_param = "l"
    elif C is None:
        validating_param = "C"
    elif svrg_m_coef is None:
        validating_param = "svrg_m_coef"
    else:
        raise
    
    if validating_param == "eta":
        my_GD_template = lambda eta: lambda f, dim: GD(f, dim, eta=eta, max_epochs=max_epochs, precision=precision)
        my_SGD_template = lambda eta: lambda f, dim: SGD(f, dim, eta=eta, max_epochs=max_epochs, precision=precision, keep_gradient=keep_gradient)
        my_SVRG_template = lambda eta: lambda f, dim: SVRG(f, dim, m_coef=svrg_m_coef, eta=eta, max_epochs=max_epochs, precision=precision)
        my_SAGA_template = lambda eta: lambda f, dim: SAGA(f, dim, eta=eta, max_epochs=max_epochs, precision=precision, keep_gradient=keep_gradient)
        my_SDCA_template = lambda X, Y, _: lambda: SDCA(X=X, Y=Y, C=C, max_epochs=max_epochs, precision=precision, keep_Q=False, keep_gradient=keep_gradient)
    else:
        my_GD_template = lambda _: lambda f, dim: GD(f, dim, eta=eta, max_epochs=max_epochs, precision=precision)
        my_SGD_template = lambda _: lambda f, dim: SGD(f, dim, eta=eta, max_epochs=max_epochs, precision=precision, keep_gradient=keep_gradient)
        my_SAGA_template = lambda _: lambda f, dim: SAGA(f, dim, eta=eta, max_epochs=max_epochs, precision=precision, keep_gradient=keep_gradient)
        if validating_param == "C":
            my_SDCA_template = lambda X, Y, C: lambda: SDCA(X=X, Y=Y, C=C, max_epochs=max_epochs, precision=precision, keep_Q=False, keep_gradient=keep_gradient)
        else:
            my_SDCA_template = lambda X, Y, _: lambda: SDCA(X=X, Y=Y, C=C, max_epochs=max_epochs, precision=precision, keep_Q=False, keep_gradient=keep_gradient)
        if validating_param == "svrg_m_coef":
            my_SVRG_template = lambda svrg_m_coef: lambda f, dim: SVRG(f, dim, m_coef=svrg_m_coef, eta=eta, max_epochs=max_epochs, precision=precision)
        else:
            my_SVRG_template = lambda _: lambda f, dim: SVRG(f, dim, m_coef=svrg_m_coef, eta=eta, max_epochs=max_epochs, precision=precision)

    method_templates = {"gd": my_GD_template, "sgd": my_SGD_template, "svrg": my_SVRG_template, "saga": my_SAGA_template, "sdca": my_SDCA_template}
    method = get_method_from_user_input()
    if method == "sdca":
        solver_template = lambda X, Y, v: SDCASolver(X, Y, my_SDCA_template(X, Y, v))
    else:
        if validating_param == "l":
            solver_template = lambda X, Y, l: LogisticRegressionSolver(X, Y, method_templates[method](l), l=l)
        else:
            solver_template = lambda X, Y, v: LogisticRegressionSolver(X, Y, method_templates[method](v), l=l)
    return solver_template, validating_param


def complete_cross_validation(
        X: np.ndarray,
        Y: np.ndarray,
        K: int,
        max_epochs: int,
        precision: float,
        keep_gradient: bool,
        param_values: list,
        *,
        eta: float=None,
        l: float=None,
        C: float=None,
        svrg_m_coef: int=None,
) -> Solver:
    """
    Select a value for a hyperparameter via cross validation, then create a solver with that parameter and train it on the whole dataset.
    For precisions on how the hyperparameter selection happens, refer to the docstring of the function get_solver_problem.
    As in get_solver_problem, the hyperparameters must be given as keyword arguments to the function.
    """
    solver_template, param_name = get_logistic_regression_solver_template(max_epochs, precision, keep_gradient, eta=eta, l=l, C=C, svrg_m_coef=svrg_m_coef)
    selected_v = cross_validation(X, Y, K, solver_template, param_values, param_name)
    solver: Solver = solver_template(X, Y, selected_v)
    return solver


def cross_validate_eta(
        X: np.ndarray,
        Y: np.ndarray,
        K: int,
        max_epochs: int,
        precision: float,
        keep_gradient: bool,
        param_values: list,
        l: float,
        C: float,
        svrg_m_coef: int,
):
    return complete_cross_validation(X, Y, K, max_epochs, precision, keep_gradient, param_values, eta=None, l=l, C=C, svrg_m_coef=svrg_m_coef)


def cross_validate_l(
        X: np.ndarray,
        Y: np.ndarray,
        K: int,
        max_epochs: int,
        precision: float,
        keep_gradient: bool,
        param_values: list,
        eta: float,
        C: float,
        svrg_m_coef: int,
):
    return complete_cross_validation(X, Y, K, max_epochs, precision, keep_gradient, param_values, eta=eta, l=None, C=C, svrg_m_coef=svrg_m_coef)


def cross_validate_C(
        X: np.ndarray,
        Y: np.ndarray,
        K: int,
        max_epochs: int,
        precision: float,
        keep_gradient: bool,
        param_values: list,
        eta: float,
        l: float,
        svrg_m_coef: int,
):
    return complete_cross_validation(X, Y, K, max_epochs, precision, keep_gradient, param_values, eta=eta, l=l, C=None, svrg_m_coef=svrg_m_coef)


def cross_validate_svrg_m_coef(
        X: np.ndarray,
        Y: np.ndarray,
        K: int,
        max_epochs: int,
        precision: float,
        keep_gradient: bool,
        param_values: list,
        eta: float,
        l: float,
        C: float,
):
    return complete_cross_validation(X, Y, K, max_epochs, precision, keep_gradient, param_values, eta=eta, l=l, C=C, svrg_m_coef=None)
