from typing import Callable, List
import numpy as np
from . import utils


def bfgs(
        function: Callable[[List[np.float64]], np.float64],
        x: List[np.float64],
        grad_method: str = 'forward_diff',
        h: np.float64 = 1e-3,
        eps: np.float64 = 1e-3,
        c_1: np.float64 = 1e-3,
        c_2: np.float64 = 0.9,
        max_iter: int = 500
        ) -> List[np.float64]:
    """
    BFGS optimization algorithm for finding the minimum of a function.

    The BFGS algorithm is an iterative method for solving unconstrained nonlinear optimization problems.
    It utilizes an approximation to the Hessian matrix and is particularly useful for functions that are 
    smooth and continuous.

    Args:
    -----
    function : Callable[[List[np.float64]], np.float64]
        The objective function to minimize. It should take a list of floats as input and return a float.
    x : List[np.float64]
        Initial guess for the parameters to optimize.
    grad_method : str, optional
        The method to use for gradient estimation ('forward_diff' or 'central_diff'). Default is 'forward_diff'.
    h : np.float64, optional
        Step size for gradient estimation. Default is 1e-3.
    eps : np.float64, optional
        Convergence threshold for the gradient. Default is 1e-3.
    c_1 : np.float64, optional
        Parameter for the backtracking line search. Default is 1e-3.
    c_2 : np.float64, optional
        Parameter for the backtracking line search. Default is 0.9.
    max_iter : int, optional
        Maximum number of iterations. Default is 500.

    Returns:
    --------
    List[np.float64]
        The optimized parameters after convergence.
    """

    if grad_method == 'forward_diff':
        grad = utils.forward_diff
    if grad_method == 'central_diff':
        grad = utils.central_diff

    I = np.eye(len(x))
    Hk = I
    for i in range(max_iter):
        grad_x = grad(function, x, h)
        if np.linalg.norm(grad_x) < eps:
            break

        p = -Hk @ grad_x
        a = utils.linear_search(function, x, grad, p, c_1, c_2)

        x_new = x + a * p
        sk = x_new - x
        x = x_new
        grad_x_new = grad(function, x, h)
        yk = grad_x_new - grad_x
        rho = 1.0 / np.dot(yk.T, sk)

        if rho <= 0:
            Hk = I
        else:
            term1 = (I - rho * np.outer(sk, yk)) @ Hk @ (I - rho * np.outer(yk, sk))
            term2 = rho * np.outer(sk, sk)
            Hk = term1 + term2

    return x


def lbfgs(
        function: Callable[[List[np.float64]], np.float64],
        x: List[np.float64],
        grad_method: str = 'forward_diff',
        h: np.float64 = 1e-3,
        eps: np.float64 = 1e-4,
        c_1: np.float64 = 1e-4,
        c_2: np.float64 = 0.9,
        max_iter: int = 1000,
        memory=20
        ) -> List[np.float64]:
    """
    L-BFGS optimization algorithm for finding the minimum of a function.

    L-BFGS (Limited-memory BFGS) is a variant of the BFGS algorithm that uses a limited amount of 
    memory to approximate the inverse Hessian. It is suitable for problems with a large number of 
    variables.

    Args:
    -----
    function : Callable[[List[np.float64]], np.float64]
        The objective function to minimize. It should take a list of floats as input and return a float.
    x : List[np.float64]
        Initial guess for the parameters to optimize.
    grad_method : str, optional
        The method to use for gradient estimation ('forward_diff' or 'central_diff'). Default is 'forward_diff'.
    h : np.float64, optional
        Step size for gradient estimation. Default is 1e-3.
    eps : np.float64, optional
        Convergence threshold for the gradient. Default is 1e-4.
    c_1 : np.float64, optional
        Parameter for the backtracking line search. Default is 1e-4.
    c_2 : np.float64, optional
        Parameter for the backtracking line search. Default is 0.9.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    memory : int, optional
        The number of previous updates to store. Default is 20.

    Returns:
    --------
    List[np.float64]
        The optimized parameters after convergence.
    """

    if grad_method == 'forward_diff':
        grad = utils.forward_diff
    if grad_method == 'central_diff':
        grad = utils.central_diff

    s_list = []
    y_list = []
    rho_list = []
    I = np.eye(len(x))
    q = grad(function, x, h)

    for i in range(max_iter):
        grad_x = grad(function, x, h)
        if np.linalg.norm(grad_x) < eps:
            return x

        q = grad_x.copy()
        a = []

        for rho, s, y in zip(reversed(rho_list), reversed(s_list), reversed(y_list)):
            alpha = rho * np.dot(s, q)
            a.append(alpha)
            q = q - alpha * y

        if len(s_list) > 0:
            gamma = np.dot(s_list[-1], y_list[-1]) / np.dot(y_list[-1], y_list[-1])
            Hk = gamma * I
        else:
            Hk = I

        p = Hk @ q

        for s, y, rho, alpha in zip(s_list, y_list, rho_list, reversed(a)):
            beta = rho * np.dot(y, p)
            p = p + s * (alpha - beta)

        p = -p

        a_k = utils.linear_search(function, x, grad, p, c_1, c_2)
        x_new = x + a_k * p

        s_k = x_new - x 
        y_k = grad(function, x_new, h) - grad_x

        if np.dot(y_k, s_k) == 0:
            return x

        rho_k = 1.0 / np.dot(y_k, s_k)

        if len(s_list) == memory:
            s_list.pop(0)
            y_list.pop(0)
            rho_list.pop(0)

        s_list.append(s_k)
        y_list.append(y_k)
        rho_list.append(rho_k)

        x = x_new

    return x

def adam(
        function: Callable[[np.ndarray], np.float64],
        x: np.ndarray,
        grad_method: str = 'forward_diff',
        h: float = 1e-5,
        alpha: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-15,
        max_iter: int = 10000
    ) -> np.ndarray:
    """
    Adam optimization algorithm for finding the minimum of a function.

    Adam (Adaptive Moment Estimation) is an optimization algorithm that computes adaptive learning rates 
    for each parameter from estimates of first and second moments of the gradients. It is suitable for 
    training deep learning models.

    Args:
    -----
    function : Callable[[np.ndarray], np.float64]
        The objective function to minimize. It should take an ndarray as input and return a float.
    x : np.ndarray
        Initial guess for the parameters to optimize.
    grad_method : str, optional
        The method to use for gradient estimation ('forward_diff' or 'central_diff'). Default is 'forward_diff'.
    h : float, optional
        Step size for gradient estimation. Default is 1e-5.
    alpha : float, optional
        Learning rate. Default is 0.001.
    beta1 : float, optional
        Exponential decay rate for the first moment estimate. Default is 0.9.
    beta2 : float, optional
        Exponential decay rate for the second moment estimate. Default is 0.999.
    eps : float, optional
        Small value to prevent division by zero. Default is 1e-15.
    max_iter : int, optional
        Maximum number of iterations. Default is 10000.

    Returns:
    --------
    np.ndarray
        The optimized parameters after convergence.
    """
    
    if grad_method == 'forward_diff':
        grad = utils.forward_diff
    elif grad_method == 'central_diff':
        grad = utils.central_diff
    else:
        raise ValueError("Unknown grad_method. Choose 'forward_diff' or 'central_diff'.")

    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0

    for i in range(max_iter):
        t += 1

        grad_x = grad(function, x, h)

        m = beta1 * m + (1 - beta1) * grad_x
        v = beta2 * v + (1 - beta2) * (grad_x ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        x = x - alpha * m_hat / (np.sqrt(v_hat) + eps)

        if np.linalg.norm(grad_x) < eps:
            print(f"Optimization converged at iteration {i}")
            break

    return x
