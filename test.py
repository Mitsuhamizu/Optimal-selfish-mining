import mdptoolbox
import mdptoolbox.example
import numpy as np
from scipy.sparse import coo_matrix as sparse_coo
from scipy.sparse import csr_matrix as sparse_csr

values = [1, 2, 3, 4, 6]
rows = [0, 1, 2, 3, 0]
cols = [1, 3, 2, 0, 1]
A = sparse_coo((values, (rows, cols)), shape=[4, 4])
print(A)
