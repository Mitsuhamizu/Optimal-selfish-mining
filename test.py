import mdptoolbox
import mdptoolbox.example
import numpy as np
from scipy.sparse import csr_matrix as sparse

a = np.zeros([2, 2])

a[0][0] = 1
a = sparse(a)

print(a[0][0])
print(type(a))
print(a)
