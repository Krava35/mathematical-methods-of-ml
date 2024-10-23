import numpy as np
from matrix import Matrix


# A = np.random.random((3, 3))
# A = A.transpose()*A

# print(np.linalg.eig(A)[1])
# Matrix_A = Matrix(A)

# print(Matrix_A.eigen()[1])

# A = np.array([[1, 3, 4], [2, 0, 9]])
# matrx_A = Matrix(A)
# Q, R = matrx_A.QR_Decomposition()
# print(Q, R)
# print(np.linalg.qr(A))

A = np.random.random((2, 3))
Matrix_A = Matrix(A)
# Matrix_A = Matrix_A*Matrix_A.transpose()
# print(Matrix_A.eigen())

# A = np.array([[4,2,0],[1,5,6]])
# Matrix_A = Matrix(A)
# Matrix_A = Matrix_A*Matrix_A.transpose()
# print(Matrix_A.eigen())

print(Matrix_A.svd())