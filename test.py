import mdptoolbox
import mdptoolbox.example
import numpy as np
from scipy.sparse import csr_matrix as sparse

P = np.zeros([2, 2])
P[0][0] = 1

P = 2*P
print(P)
