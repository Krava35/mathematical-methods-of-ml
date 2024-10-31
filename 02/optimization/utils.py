from typing import Callable, List
import numpy as np
import numpy.typing as npt


def forward_diff(
        function: Callable[[List[np.float64]], np.float64],
        x: List[np.float64],
        h: np.float64 = 1e-3
        ) -> npt.NDArray:
    """
    Computes the gradient of a function using forward finite difference.

    This method estimates the gradient of a given function at a point using 
    the forward difference method, which is a numerical technique for 
    approximating derivatives.

    Args:
    -----
    function : Callable[[List[np.float64]], np.float64]
        The objective function for which the gradient is computed. It should take
        a list of floats as input and return a float.
    x : List[np.float64]
        The point at which to evaluate the gradient.
    h : np.float64, optional
        The step size for the finite difference. Default is 1e-3.

    Returns:
    --------
    npt.NDArray
        The estimated gradient as a NumPy array.
    """
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
    """
    Computes the gradient of a function using central finite difference.

    This method estimates the gradient of a given function at a point using 
    the central difference method, which provides a more accurate approximation
    of the derivative than the forward difference method.

    Args:
    -----
    function : Callable[[List[np.float64]], np.float64]
        The objective function for which the gradient is computed. It should take
        a list of floats as input and return a float.
    x : List[np.float64]
        The point at which to evaluate the gradient.
    h : np.float64, optional
        The step size for the finite difference. Default is 1e-3.

    Returns:
    --------
    npt.NDArray
        The estimated gradient as a NumPy array.
    """
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
    """
    Performs a backtracking line search to find an appropriate step size.

    The line search algorithm seeks a step size that sufficiently reduces the
    function value while satisfying the conditions defined by the Armijo 
    (or sufficient decrease) condition and the curvature condition.

    Args:
    -----
    function : Callable[[List[np.float64]], np.float64]
        The objective function to minimize. It should take a list of floats as 
        input and return a float.
    x : List[float]
        Current point in the parameter space.
    grad : Callable[[List[np.float64]], npt.NDArray]
        Gradient function for the objective function.
    p : npt.NDArray
        Search direction, typically the negative gradient.
    c_1 : np.float64, optional
        Parameter for the Armijo condition. Default is 1e-4.
    c_2 : np.float64, optional
        Parameter for the curvature condition. Default is 0.9.
    a_max : np.float64, optional
        Maximum step size to consider. Default is 1.0.
    max_iter : int, optional
        Maximum number of iterations for the line search. Default is 1000.

    Returns:
    --------
    np.float64
        The step size that satisfies the conditions of the line search.
    """

    def zoom(a_low, a_high):
        """
        Helper function to perform zoom phase of the line search.

        The zoom function narrows down the range of step sizes to find 
        an acceptable one that satisfies both the Armijo and curvature conditions.

        Args:
        -----
        a_low : float
            The lower bound of the step size.
        a_high : float
            The upper bound of the step size.

        Returns:
        --------
        float
            The selected step size that meets the conditions.
        """
        i = 1
        while i < 1000:
            a_j = 0.5 * (a_low + a_high)
            x_new = x + a_j * p
            condition_1 = function(x_new) > function(x) + c_1 * a_j * np.dot(grad(function, x), p)
            condition_2 = function(x_new) >= function(x + a_low * p)
            if condition_1 or condition_2:
                a_high = a_j
            else:
                if np.abs(np.dot(grad(function, x_new), p)) <= -c_2 * np.dot(grad(function, x), p):
                    return a_j
                if np.dot(grad(function, x_new), p) >= 0:
                    a_high = a_j
                else:
                    a_low = a_j
            i += 1

    a = [0, np.random.uniform(0, a_max)]
    i = 1
    while i <= max_iter:
        print(type(x), x)
        print(type(p), p)
        x_new = x + a[1] * p
        condition_1 = function(x_new) > function(x) + c_1 * a[1] * grad(function, x) @ p
        condition_2 = function(x_new) >= function(x + a[0] * p) and i > 1
        if condition_1 or condition_2:
            return zoom(a[0], a[1])

        if np.abs(np.dot(grad(function, x_new), p)) <= -c_2 * np.dot(grad(function, x), p):
            return a[1]

        if np.dot(grad(function, x_new), p) >= 0:
            return zoom(a[0], a[1])

        a = [a[1], np.random.uniform(a[1], a_max)]
        i += 1

    return a[1]
