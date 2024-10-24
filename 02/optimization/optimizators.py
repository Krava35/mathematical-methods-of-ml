from typing import Callable, List
import numpy as np
from . import utils


def bfgs(
        function: Callable[[List[np.float64]], np.float64],
        x: List[np.float64],
        grad_method: str = 'forward_diff',
        h: np.float64 = 1e-3,
        eps: np.float64 = 1e-4,
        c_1: np.float64 = 1e-4,
        c_2: np.float64 = 0.9,
        max_iter: int = 1000
        ):

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
        a = utils.linear_search(function, grad, x, p, c_1, c_2)

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
        ):

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
            a.append(rho * np.dot(s, q))
            q -= a * y

        if len(s_list) > 0:
            gamma = np.dot(s_list[-1], y_list[-1]) / np.dot(y_list[-1], y_list[-1])
            Hk = gamma * I
        else:
            Hk = I

        p = Hk @ q
        for s, y, rho, a in zip(s_list, y_list, rho_list, reversed(a)):
            beta = rho * np.dot(y, p)
            p += s * (a - beta)

        p = -p
        a_k = utils.linear_search(function, grad, x, p, c_1, c_2)
        x_new = x + a_k * p
        s_k = x_new - x
        y_k = grad(function, x_new, h) - grad_x
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
