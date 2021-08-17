import mdptoolbox
import mdptoolbox.example
import numpy as np
from scipy.sparse import csr_matrix as sparse

D = 5
n = 10
p = 0.1
rho = 0.3
result1 = (1 - pow((1-n*p), 5)) * rho
result2 = 0
for i in range(0, D):
    result2 += pow(1-n*p, i)*rho
print(result1)
print(result2)
P = np.zeros([2, 2])
P = sparse(P)
P[0:2, 0] = 1
print(P)
