# Mathematical Methods of ML

This repository contains implementations of various linear algebra techniques and optimization algorithms, developed as part of two lab assignments. The focus of the first lab was on Singular Value Decomposition (SVD), eigenvalues, and eigenvectors. The second lab concentrated on implementing optimization algorithms including BFGS, L-BFGS, Adam, finite difference methods and linear search.

## Lab 1: PCA with Singular Value Decomposition (SVD)

In the first lab, we explored the following concepts:

- **Singular Value Decomposition (SVD)**: A mathematical technique that decomposes a matrix into three other matrices, capturing essential properties of the original matrix. The function computes the matrices U, Sigma, and V^T, which represent the singular vectors and singular values of the input matrix.

- **Eigenvalues and Eigenvectors**: These are fundamental concepts in linear algebra that provide insights into the properties of a linear transformation. The implementation calculates the eigenvalues and eigenvectors of a given matrix, helping to understand its behavior and characteristics.

- **Principal Component Analysis (PCA)**: PCA is a dimensionality reduction technique that utilizes SVD. It identifies the directions (principal components) along which the variation of the data is maximized. By projecting the data onto these principal components, PCA reduces the number of dimensions while retaining the most important information. This is particularly useful in data analysis and visualization, where it simplifies complex datasets.
## Lab 2: Optimization Algorithms

In the second lab, we implemented several optimization algorithms designed to minimize functions, including:

- **BFGS (Broyden–Fletcher–Goldfarb–Shanno algorithm)**: An iterative method for solving unconstrained nonlinear optimization problems. It uses an approximation to the Hessian matrix to find search directions.

- **L-BFGS (Limited-memory BFGS)**: A variant of BFGS that is designed for optimization problems with a large number of variables. It maintains a limited amount of information to update the approximate Hessian.

- **Adam (Adaptive Moment Estimation)**: An optimization algorithm that computes adaptive learning rates for each parameter, combining ideas from both momentum and RMSprop.

- **Finite Differences**: Methods for estimating the gradient of a function, allowing for numerical optimization when analytical gradients are not available
