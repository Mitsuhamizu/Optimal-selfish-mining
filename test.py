import mdptoolbox
import mdptoolbox.example
import numpy as np
from scipy.sparse import csr_matrix as sparse

P = np.zeros([2, 2])
P[0][0] = 1

R = np.zeros([2, 2])
R[0][0] = 1
P = sparse(P)
R = sparse(R)
result = 2*P.multiply(3*R)
print(result)
