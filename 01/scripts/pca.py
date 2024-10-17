import numpy as np
import tqdm
from matrix import Matrix


class PCA:
    def __init__(self, k, kernel: str = 'linear'):
        self.k  = k
        self.kernel = kernel
    
    def fit(self) -> Matrix:
        pass
