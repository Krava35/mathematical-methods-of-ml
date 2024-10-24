from typing import Callable, List
import numpy as np
import numpy.typing as npt


def forward_diff(
        function: Callable[[List[np.float64]], np.float64],
        x: List[np.float64],
        h: np.float64 = 1e-3
        ) -> npt.NDArray:
    n = len(x)
    grad = np.zeros(n)

    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        grad[i] = (function(x + h * e) - function(x)) / h

    return grad


def central_diff(
        function: Callable[[List[np.float64]], np.float64],
        x: List[np.float64],
        h: np.float64 = 1e-3
        ) -> npt.NDArray:
    n = len(x)

    grad = np.zeros(n)
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        grad[i] = (function(x + h * e) - function(x - h * e)) / (2 * h)

    return grad


def linear_search(
        function: Callable[[List[np.float64]], np.float64],
        x: List[float],
        grad: Callable[[List[np.float64]], npt.NDArray],
        p: npt.NDArray,
        c_1: np.float64 = 1e-4,
        c_2: np.float64 = 0.9,
        a_max: np.float64 = 1.0,
        max_iter: int = 1000
        ) -> np.float64:

    def zoom(a_low, a_high):
        i = 1
        while i < 1000:
            a_j = 0.5 * (a_low + a_high)
            x_new = x + a_j * p
            condition_1 = function(x_new) > function(x) + c_1 * a_j * np.dot(grad(x), p)
            condition_2 = function(x_new) >= function(x + a_low * p)
            if condition_1 or condition_2:
                a_high = a_j
            else:
                if np.abs(np.dot(grad(x_new), p)) <= -c_2 * np.dot(grad(x), p):
                    return a_j
                if np.dot(grad(x_new), p) >= 0:
                    a_high = a_j
                else:
                    a_low = a_j
            i += 1

    a = [0, np.random.uniform(0, a_max)]
    i = 1
    while i <= max_iter:
        x_new = x + a[1] * p
        condition_1 = function(x_new) > function(x) + c_1 * a[1] * grad(x) @ p
        condition_2 = function(x_new) >= function(x + a[0] * p) and i > 1
        if condition_1 or condition_2:
            return zoom(a[0], a[1])

        if np.abs(np.dot(grad(x_new), p)) <= -c_2 * np.dot(grad(x), p):
            return a[1]

        if np.dot(grad(x_new), p) >= 0:
            return zoom(a[0], a[1])

        a = [a[1], np.random.uniform(a[1], a_max)]
        i += 1

    return a[1]
