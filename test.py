import mdptoolbox
import mdptoolbox.example
import numpy as np
from scipy.sparse import coo_matrix as sparse_coo
from scipy.sparse import csr_matrix as sparse_csr

a = np.array([np.array([[0, 1], [0, 2]]), np.array([[0, 1], [0, 2]])])
print(a[0, 0, 0])
