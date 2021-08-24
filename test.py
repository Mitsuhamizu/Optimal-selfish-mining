import mdptoolbox
import mdptoolbox.example
import numpy as np
from scipy.sparse import coo_matrix as sparse_coo
from scipy.sparse import csr_matrix as sparse_csr

length = 5
A = np.ones([1, length])
B = np.ones([length, 1])
print(A - B)
# print(A)
