import numpy as np
from matrix import Matrix


A = np.random.random((3, 3))
A = A.transpose()*A

print(np.linalg.eig(A)[1])
Matrix_A = Matrix(A)

print(Matrix_A.eigen()[1])

A = np.array([[4,2,0],[1,5,6]])
Matrix_A = Matrix(A)

print(Matrix_A.svd())