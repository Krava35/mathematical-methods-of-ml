from linear.matrix import Matrix
import numpy as np
import unittest

class TestMatrix(unittest.TestCase):
    
    def test_addition(self):
        m1 = Matrix(np.array([[1, 2], [3, 4]]))
        m2 = Matrix(np.array([[5, 6], [7, 8]]))
        expected = np.array([[6, 8], [10, 12]])
        result = (m1 + m2).value
        np.testing.assert_array_equal(result, expected)

    def test_subtraction(self):
        m1 = Matrix(np.array([[1, 2], [3, 4]]))
        m2 = Matrix(np.array([[5, 6], [7, 8]]))
        expected = np.array([[-4, -4], [-4, -4]])
        result = (m1 - m2).value
        np.testing.assert_array_equal(result, expected)

    def test_multiplication(self):
        m3 = Matrix(np.array([[1, 2, 3], [4, 5, 6]]))
        m4 = Matrix(np.array([[7, 8], [9, 10], [11, 12]]))
        expected = np.array([[58, 64], [139, 154]])
        result = (m3 * m4).value
        np.testing.assert_array_equal(result, expected)

    def test_addition_mismatched_dimensions(self):
        m1 = Matrix(np.array([[1, 2], [3, 4]]))
        m3 = Matrix(np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]]))
        with self.assertRaises(ValueError) as context:
            _ = m1 + m3
    
        # Check if the error message is what we expect
        self.assertEqual(str(context.exception), "Can't add matrix with shapes: (2, 2), (3, 3)")

    def test_multiplication_mismatched_dimensions(self):
        m1 = Matrix(np.array([[1, 2], [3, 4]]))
        m2 = Matrix(np.array([[5, 6], [7, 8], [3, 4]]))
        with self.assertRaises(ValueError) as context:
            _ = m1 * m2
        self.assertEqual(str(context.exception), "Can't multiply matrix with shapes: (2, 2), (3, 2)")

if __name__ == "__main__":
    unittest.main()